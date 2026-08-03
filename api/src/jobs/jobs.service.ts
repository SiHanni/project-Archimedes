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
import {
  SINGLE_IMAGE_FIELD,
  type CaptureMode,
  type CreateJobFormFields,
  type JobUploadFiles,
} from './jobs.types';

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

  /** 파일 1개를 S3 에 올리고 key 를 돌려준다. MIME 검증 포함. */
  private async putImage(
    jobId: string,
    field: string,
    file: Express.Multer.File,
  ): Promise<string> {
    const mime = file.mimetype || '';
    if (!ALLOWED_IMAGE_MIME.test(mime)) {
      throw new BadRequestException(
        `Field "${field}": allowed image/jpeg, image/png, image/webp (got ${mime || 'empty'})`,
      );
    }
    const ext = mime.includes('png')
      ? 'png'
      : mime.includes('webp')
        ? 'webp'
        : 'jpg';
    const key = `uploads/${jobId}/${field}.${ext}`;
    await this.s3.send(
      new PutObjectCommand({
        Bucket: this.bucket,
        Key: key,
        Body: file.buffer,
        ContentType: mime || 'image/jpeg',
      }),
    );
    return key;
  }

  async createFromUpload(
    files: JobUploadFiles,
    captureMode: CaptureMode,
    body: CreateJobFormFields,
  ) {
    const jobId = uuidv4();

    // worker `JobInputRecord` 와 동일한 형태로 저장한다.
    // 단일 모드는 image, 다뷰 모드는 views 만 채운다.
    let image: string | null = null;
    let views: Record<string, string> | null = null;

    if (captureMode === 'multiview') {
      views = {};
      for (const v of VIEW_KEYS) {
        views[v] = await this.putImage(jobId, v, files[v]![0]);
      }
    } else {
      image = await this.putImage(
        jobId,
        SINGLE_IMAGE_FIELD,
        files[SINGLE_IMAGE_FIELD]![0],
      );
    }

    let ref: number | null = null;
    if (body.reference_weight_g != null && body.reference_weight_g !== '') {
      const n = parseFloat(String(body.reference_weight_g));
      ref = Number.isFinite(n) ? n : null;
    }
    const input = {
      capture_mode: captureMode,
      image,
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
    return {
      id: r.id,
      status: r.status,
      algorithmVersion: r.algorithm_version,
      input: parseJson(r.input_json),
      result: parseJson(r.result_json),
      error:
        r.error_code != null
          ? { code: r.error_code, message: r.error_message }
          : null,
      createdAt: r.created_at,
      updatedAt: r.updated_at,
    };
  }

  async upsertMassFeedback(jobId: string, actualMassG: number, notes: string | null) {
    const row = await this.getById(jobId);
    if (!row) throw new NotFoundException('Job not found');
    if (row.status !== 'completed') {
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
