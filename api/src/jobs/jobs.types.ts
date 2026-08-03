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

export type CaptureMode = 'single' | 'multiview';

export type CreateJobFormFields = {
  metal?: string;
  purity?: string;
  product_k?: string;
  reference_weight_g?: string;
  knows_weight?: string;
  capture_mode?: string;
};
