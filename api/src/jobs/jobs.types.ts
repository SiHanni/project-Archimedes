import type { ViewKey } from '../common/views';

/** 단일사진 업로드 필드명 — worker `runner.SINGLE_VIEW_KEY` 와 대응 */
export const SINGLE_IMAGE_FIELD = 'image' as const;

/**
 * Multer FileFields 결과.
 *
 * v2 는 `image` 1장이 기본이고, 5뷰(`front`…`back`)는 고신뢰 옵션 모드다.
 * 둘 중 하나만 채워져 오므로 전부 optional 이다.
 */
export type JobUploadFiles = Partial<
  Record<ViewKey | typeof SINGLE_IMAGE_FIELD, Express.Multer.File[]>
>;

// 에라토스테네스(기준물 없음) 두 모드 — 비용이 크게 달라 일부러 나눴다.
//   outline  : 누끼만. 몇 초.
//   distance : 누끼 + 거리. Depth Pro 를 돌려 2~3분.
export type CaptureMode = 'single' | 'multiview' | 'outline' | 'distance';

export type CreateJobFormFields = {
  metal?: string;
  purity?: string;
  product_k?: string;
  reference_weight_g?: string;
  reference_thickness_mm?: string;
  declared_gold_g?: string;
  knows_weight?: string;
  capture_mode?: string;
  /** 거리 추정에서 물체의 실제 긴 변(mm). 알면 사전값보다 정확하다. */
  known_long_mm?: string;
};
