import {
  BadRequestException,
  Controller,
  Get,
  Header,
  NotFoundException,
  Param,
  ParseUUIDPipe,
  Res,
} from '@nestjs/common';
import type { Response } from 'express';
import { JobsService } from './jobs.service';

/**
 * 세그멘테이션 산출물(누끼 오버레이·마스크·컷아웃)을 브라우저에 준다.
 *
 * S3(MinIO)를 직접 노출하지 않고 **API 가 프록시**한다. 로컬 MinIO 는 내부
 * 주소(`http://minio:9000`)라 브라우저·터널에서 닿지 않고, 프리사인 URL 을 쓰면
 * 터널 밖 호스트가 새어 나간다.
 */
const ALLOWED = {
  'overlay.jpg': 'image/jpeg',
  'mask.png': 'image/png',
  'cutout.png': 'image/png',
} as const;

type AssetName = keyof typeof ALLOWED;

@Controller('jobs')
export class JobAssetsController {
  constructor(private readonly jobs: JobsService) {}

  @Get(':id/assets/:name')
  @Header('Cache-Control', 'public, max-age=3600')
  async asset(
    @Param('id', new ParseUUIDPipe({ version: '4' })) id: string,
    @Param('name') name: string,
    @Res() res: Response,
  ) {
    // 화이트리스트로만 접근 — 경로 조작으로 다른 객체를 읽지 못하게
    if (!(name in ALLOWED)) {
      throw new BadRequestException(
        `Unknown asset "${name}". allowed=${Object.keys(ALLOWED).join(', ')}`,
      );
    }
    const body = await this.jobs.readSegmentationAsset(id, name as AssetName);
    if (!body) throw new NotFoundException('Asset not found');

    res.setHeader('Content-Type', ALLOWED[name as AssetName]);
    res.send(body);
  }
}
