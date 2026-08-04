import {
  Body,
  Controller,
  Get,
  Param,
  ParseUUIDPipe,
  Post,
  UploadedFiles,
  UseInterceptors,
  BadRequestException,
  NotFoundException,
} from '@nestjs/common';
import { FileFieldsInterceptor } from '@nestjs/platform-express';
import { memoryStorage } from 'multer';
import { VIEW_KEYS } from '../common/views';
import { JobsService } from './jobs.service';
import {
  SINGLE_IMAGE_FIELD,
  type CaptureMode,
  type CreateJobFormFields,
  type JobUploadFiles,
} from './jobs.types';

const upload = memoryStorage();
const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;

/** v2 기본은 `image` 1장, 5뷰는 고신뢰 옵션 모드 */
const fileFields = [
  { name: SINGLE_IMAGE_FIELD, maxCount: 1 },
  ...VIEW_KEYS.map((name) => ({ name, maxCount: 1 })),
];

@Controller('jobs')
export class JobsController {
  constructor(private readonly jobs: JobsService) {}

  @Post()
  @UseInterceptors(
    FileFieldsInterceptor(fileFields, {
      storage: upload,
      limits: { fileSize: MAX_UPLOAD_BYTES },
    }),
  )
  async create(
    @UploadedFiles() files: JobUploadFiles,
    @Body() body: CreateJobFormFields,
  ) {
    const mode = this.resolveCaptureMode(files, body.capture_mode);
    if (mode === 'multiview') {
      const missing = VIEW_KEYS.filter((v) => !files[v]?.[0]);
      if (missing.length) {
        throw new BadRequestException(
          `capture_mode=multiview requires all 5 views. missing: ${missing.join(', ')}`,
        );
      }
    } else if (!files[SINGLE_IMAGE_FIELD]?.[0]) {
      throw new BadRequestException(
        `capture_mode=single requires file field "${SINGLE_IMAGE_FIELD}"`,
      );
    }

    return this.jobs.createFromUpload(files, mode, {
      metal: body.metal || 'gold',
      purity: body.purity || '18k',
      product_k: body.product_k || 'ring',
      reference_weight_g: body.reference_weight_g,
      reference_thickness_mm: body.reference_thickness_mm,
      knows_weight: body.knows_weight,
    });
  }

  /**
   * 명시된 capture_mode 를 우선하고, 없으면 올라온 파일로 추론한다.
   * (구버전 클라이언트는 capture_mode 없이 5뷰만 보낸다)
   */
  private resolveCaptureMode(
    files: JobUploadFiles,
    requested?: string,
  ): CaptureMode {
    const asked = (requested || '').trim().toLowerCase();
    if (asked === 'single' || asked === 'multiview') return asked;
    if (files[SINGLE_IMAGE_FIELD]?.[0]) return 'single';
    if (VIEW_KEYS.every((v) => files[v]?.[0])) return 'multiview';
    throw new BadRequestException(
      `Upload either "${SINGLE_IMAGE_FIELD}" (single photo) or all of: ${VIEW_KEYS.join(', ')}`,
    );
  }

  /** 실측 무게 등록(캘리브레이션용). job 완료 후에만. */
  @Post(':id/feedback')
  async postFeedback(
    @Param('id', new ParseUUIDPipe({ version: '4' })) id: string,
    @Body() body: { actualMassG?: number; notes?: string },
  ) {
    const m = body.actualMassG;
    if (m == null || typeof m !== 'number' || !Number.isFinite(m) || m <= 0) {
      throw new BadRequestException('actualMassG must be a positive number');
    }
    return this.jobs.upsertMassFeedback(id, m, body.notes ?? null);
  }

  @Get(':id')
  async get(@Param('id', new ParseUUIDPipe({ version: '4' })) id: string) {
    const row = await this.jobs.getById(id);
    if (!row) throw new NotFoundException('Job not found');
    return row;
  }
}
