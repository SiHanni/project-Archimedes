import {
  BadRequestException,
  Injectable,
  NotFoundException,
  OnModuleDestroy,
  OnModuleInit,
} from '@nestjs/common';
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
import { createPool, Pool, RowDataPacket } from 'mysql2/promise';
import Redis from 'ioredis';
import { v4 as uuidv4 } from 'uuid';
import { VIEW_KEYS } from '../common/views';
import type { CreateJobFormFields, UploadsByView } from './jobs.types';

const ALLOWED_IMAGE_MIME = /^image\/(jpeg|jpg|png|webp)$/i;

@Injectable()
export class JobsService implements OnModuleInit, OnModuleDestroy {
  private pool!: Pool;
  private redis!: Redis;
  private s3!: S3Client;
  private bucket!: string;
  private queue!: string;

  onModuleInit() {
    this.pool = createPool({
      host: process.env.MYSQL_HOST || 'localhost',
      port: parseInt(process.env.MYSQL_PORT || '3306', 10),
      user: process.env.MYSQL_USER || 'archimedes',
      password: process.env.MYSQL_PASSWORD || 'archimedes',
      database: process.env.MYSQL_DATABASE || 'archimedes',
      waitForConnections: true,
      connectionLimit: 10,
    });
    this.redis = new Redis(process.env.REDIS_URL || 'redis://localhost:6379');
    this.bucket = process.env.S3_BUCKET || 'archimedes-uploads';
    this.queue = process.env.QUEUE_NAME || 'archimedes:queue';
    this.s3 = new S3Client({
      region: process.env.S3_REGION || 'us-east-1',
      endpoint: process.env.S3_ENDPOINT,
      credentials: {
        accessKeyId: process.env.S3_ACCESS_KEY || 'minioadmin',
        secretAccessKey: process.env.S3_SECRET_KEY || 'minioadmin',
      },
      forcePathStyle: true,
    });
  }

  async onModuleDestroy() {
    await this.pool?.end();
    this.redis?.disconnect();
  }

  async createFromUpload(files: UploadsByView, body: CreateJobFormFields) {
    const jobId = uuidv4();
    const views: Record<string, string> = {};
    for (const v of VIEW_KEYS) {
      const file = files[v][0];
      const mime = file.mimetype || '';
      if (!ALLOWED_IMAGE_MIME.test(mime)) {
        throw new BadRequestException(
          `Field "${v}": allowed image/jpeg, image/png, image/webp (got ${mime || 'empty'})`,
        );
      }
      const ext =
        mime.includes('png') ? 'png' : mime.includes('webp') ? 'webp' : 'jpg';
      const key = `uploads/${jobId}/${v}.${ext}`;
      await this.s3.send(
        new PutObjectCommand({
          Bucket: this.bucket,
          Key: key,
          Body: file.buffer,
          ContentType: file.mimetype || 'image/jpeg',
        }),
      );
      views[v] = key;
    }
    let ref: number | null = null;
    if (body.reference_weight_g != null && body.reference_weight_g !== '') {
      const n = parseFloat(String(body.reference_weight_g));
      ref = Number.isFinite(n) ? n : null;
    }
    const input = {
      views,
      metal: (body.metal || 'gold').toLowerCase(),
      purity: (body.purity || '18k').toLowerCase(),
      product_k: (body.product_k || 'ring').toLowerCase(),
      reference_weight_g: ref,
      knows_weight: body.knows_weight || null,
    };
    await this.pool.execute(
      `INSERT INTO jobs (id, status, input_json, created_at) VALUES (?, 'pending', ?, NOW())`,
      [jobId, JSON.stringify(input)],
    );
    await this.redis.lpush(this.queue, jobId);
    return { id: jobId, status: 'pending' };
  }

  async getById(id: string) {
    const [rows] = await this.pool.query<RowDataPacket[]>(
      `SELECT id, status, algorithm_version, input_json, result_json, error_code, error_message, created_at, updated_at FROM jobs WHERE id = ?`,
      [id],
    );
    if (!rows.length) return null;
    const r = rows[0];
    const parseJson = (x: unknown) => {
      if (x == null) return null;
      if (typeof x === 'object') return x;
      try {
        return JSON.parse(String(x));
      } catch {
        return null;
      }
    };
    const parsedResult = parseJson(r.result_json);
    const resultObj = parsedResult as Record<string, unknown> | null;
    const nestedErr =
      resultObj && typeof resultObj === 'object' && resultObj.error && typeof resultObj.error === 'object'
        ? (resultObj.error as Record<string, unknown>)
        : null;
    const retryViews =
      nestedErr && Array.isArray(nestedErr.retry_views)
        ? (nestedErr.retry_views.filter((x) => typeof x === 'string') as string[])
        : [];
    const retryStep = typeof nestedErr?.retry_step === 'string' ? nestedErr.retry_step : null;
    const errorSeverity =
      nestedErr?.error_severity === 'soft' || nestedErr?.error_severity === 'hard'
        ? (nestedErr.error_severity as 'soft' | 'hard')
        : 'hard';
    const suggestedAction =
      typeof nestedErr?.suggested_action === 'string'
        ? nestedErr.suggested_action
        : retryViews.length || retryStep
          ? 'retry_one_view'
          : null;
    const mergedRetryViews = retryViews.length ? retryViews : retryStep ? [retryStep] : [];
    return {
      id: r.id,
      status: r.status,
      algorithmVersion: r.algorithm_version,
      input: parseJson(r.input_json),
      result: parsedResult,
      error:
        r.error_code != null
          ? {
              code: r.error_code,
              message: r.error_message,
              retryViews: mergedRetryViews,
              errorSeverity,
              suggestedAction,
            }
          : null,
      createdAt: r.created_at,
      updatedAt: r.updated_at,
    };
  }

  async upsertMassFeedback(jobId: string, actualMassG: number, notes: string | null) {
    const row = await this.getById(jobId);
    if (!row) throw new NotFoundException('Job not found');
    if (row.status !== 'completed' && row.status !== 'completed_low_confidence') {
      throw new BadRequestException('Job must be completed before feedback');
    }
    const res = row.result as { mass_est_g?: number } | null;
    if (!res || typeof res.mass_est_g !== 'number') {
      throw new BadRequestException('Job has no mass result to compare');
    }
    const [existing] = await this.pool.query<RowDataPacket[]>(
      `SELECT id FROM mass_feedback WHERE job_id = ? LIMIT 1`,
      [jobId],
    );
    if (existing.length) {
      await this.pool.execute(
        `UPDATE mass_feedback SET actual_mass_g = ?, notes = ? WHERE job_id = ?`,
        [actualMassG, notes, jobId],
      );
    } else {
      await this.pool.execute(
        `INSERT INTO mass_feedback (job_id, actual_mass_g, notes) VALUES (?, ?, ?)`,
        [jobId, actualMassG, notes],
      );
    }
    const ratio = actualMassG / res.mass_est_g;
    return {
      ok: true,
      jobId,
      massEstG: res.mass_est_g,
      actualMassG,
      ratioActualOverEst: Math.round(ratio * 10000) / 10000,
      message:
        '저장됨. Hollow α 튜닝 제안은 worker 스크립트 calibration_suggest.py 를 실행하세요.',
    };
  }
}
