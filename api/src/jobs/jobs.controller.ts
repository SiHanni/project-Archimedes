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
import type { CreateJobFormFields, UploadsByView } from './jobs.types';

const upload = memoryStorage();
const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;

const fileFields = VIEW_KEYS.map((name) => ({ name, maxCount: 1 }));

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
    @UploadedFiles() files: UploadsByView,
    @Body() body: CreateJobFormFields,
  ) {
    for (const v of VIEW_KEYS) {
      if (!files[v]?.[0]) {
        throw new BadRequestException(`Missing file field: ${v}`);
      }
    }
    return this.jobs.createFromUpload(files, {
      metal: body.metal || 'gold',
      purity: body.purity || '18k',
      product_k: body.product_k || 'ring',
      reference_weight_g: body.reference_weight_g,
      knows_weight: body.knows_weight,
    });
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
