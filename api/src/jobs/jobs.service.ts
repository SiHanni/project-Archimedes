import {
  BadRequestException,
  Injectable,
  NotFoundException,
  OnModuleDestroy,
  OnModuleInit,
} from '@nestjs/common';
import { S3Client, PutObjectCommand, GetObjectCommand } from '@aws-sdk/client-s3';
import { createPool, Pool, RowDataPacket } from 'mysql2/promise';
import Redis from 'ioredis';
import { v4 as uuidv4 } from 'uuid';
import { VIEW_KEYS } from '../common/views';
import { PricingService } from '../pricing/pricing.service';
import { buildQuote } from '../pricing/quote';
import {
  SINGLE_IMAGE_FIELD,
  type CaptureMode,
  type CreateJobFormFields,
  type JobUploadFiles,
} from './jobs.types';

const ALLOWED_IMAGE_MIME = /^image\/(jpeg|jpg|png|webp)$/i;

@Injectable()
export class JobsService implements OnModuleInit, OnModuleDestroy {
  constructor(private readonly pricing: PricingService) {}

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

    const num = (v: string | undefined): number | null => {
      if (v == null || v === '') return null;
      const n = parseFloat(String(v));
      return Number.isFinite(n) && n > 0 ? n : null;
    };
    const ref = num(body.reference_weight_g);
    // 골드바처럼 사진으로 못 재는 두께 — worker 가 관측 대신 이 값을 쓴다
    const refThickness = num(body.reference_thickness_mm);
    // 도금·금박 제품에 인쇄된 순금 함유량 — 부피로는 못 재므로 표기값을 쓴다
    const declaredGold = num(body.declared_gold_g);
    // 거리 추정용 실제 긴 변(mm)
    const knownLong = num(body.known_long_mm);
    const input = {
      capture_mode: captureMode,
      image,
      views,
      metal: (body.metal || 'gold').toLowerCase(),
      purity: (body.purity || '18k').toLowerCase(),
      product_k: (body.product_k || 'ring').toLowerCase(),
      reference_weight_g: ref,
      reference_thickness_mm: refThickness,
      declared_gold_g: declaredGold,
      knows_weight: body.knows_weight || null,
      // 거리 추정에서 물체의 실제 긴 변(mm). 사전값보다 정확하므로 있으면 그걸 쓴다.
      known_long_mm: knownLong,
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
    // soft 에러 job 은 result_json 에 **분석 결과가 아니라 에러 페이로드**가 들어간다.
    // 이걸 그대로 `result` 로 내보내면 클라이언트가 result 가 있다고 믿고
    // mass_est_g 같은 필드를 건드려 터진다(실제로 흰 화면 사고).
    // 분석 결과의 판별 기준은 `mass_est_g` 존재 여부다.
    // 에라토스테네스(outline)는 **일부러 무게를 내지 않는다.** 그래서 mass_est_g
    // 만으로 판별하면 정상 결과가 통째로 null 이 되어 누끼가 화면에 안 뜬다.
    const captureMode = (resultObj?.meta as Record<string, unknown> | undefined)?.capture_mode;
    const isAnalysis =
      !!resultObj &&
      (typeof (resultObj as { mass_est_g?: unknown }).mass_est_g === 'number' ||
        captureMode === 'outline');
    const workflow =
      (resultObj?.meta as Record<string, unknown> | undefined)?.workflow ?? null;
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
    const quote = isAnalysis
      ? await this.buildQuoteFor(parseJson(r.input_json), resultObj)
      : null;
    return {
      id: r.id,
      status: r.status,
      algorithmVersion: r.algorithm_version,
      input: parseJson(r.input_json),
      // 분석 결과가 아니면 null — 에러 정보는 error/workflow 로만 전달한다
      result: isAnalysis ? parsedResult : null,
      workflow,
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
      quote,
      createdAt: r.created_at,
      updatedAt: r.updated_at,
    };
  }

  /**
   * 견적은 **조회 시점**에 계산한다. 시세는 계속 변하므로 job 결과에 굳혀 두면
   * 오래된 금액이 남는다. 무게(추정)와 금액(시세)의 수명이 다르다.
   */
  private async buildQuoteFor(input: unknown, result: Record<string, unknown> | null) {
    if (!result || typeof result.mass_est_g !== 'number') return null;
    const inp = (input ?? {}) as Record<string, unknown>;
    const metal = String(inp.metal ?? 'gold').toLowerCase();
    const purity = String(inp.purity ?? '18k').toLowerCase();

    const meta = (result.meta ?? {}) as Record<string, unknown>;
    const sanity = (meta.sanity ?? {}) as Record<string, unknown>;

    const spot = await this.pricing.getSpot(metal, purity);
    return buildQuote(
      {
        massEstG: result.mass_est_g as number,
        massRange: (result.mass_range ?? null) as never,
        confidenceTier: String(result.confidence_tier ?? 'low'),
        suppressMassDisplay: sanity.suppress_mass_display === true,
      },
      spot,
      this.pricing.buyRate,
    );
  }

  /** 세그 산출물을 S3 에서 읽어 바이트로 돌려준다. 없으면 null. */
  async readSegmentationAsset(jobId: string, name: string): Promise<Buffer | null> {
    try {
      const out = await this.s3.send(
        new GetObjectCommand({
          Bucket: this.bucket,
          Key: `segmentation/${jobId}/${name}`,
        }),
      );
      const bytes = await out.Body?.transformToByteArray();
      return bytes ? Buffer.from(bytes) : null;
    } catch {
      // 아직 생성 전이거나 저장이 꺼져 있는 경우 — 404 로 다룬다
      return null;
    }
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
