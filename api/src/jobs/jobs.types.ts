import type { ViewKey } from '../common/views';

/** Multer FileFields 결과 타입 */
export type UploadsByView = Record<ViewKey, Express.Multer.File[]>;

export type CreateJobFormFields = {
  metal?: string;
  purity?: string;
  product_k?: string;
  reference_weight_g?: string;
  knows_weight?: string;
};
