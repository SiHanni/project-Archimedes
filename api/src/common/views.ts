/** 5-view keys — worker `VIEW_ORDER` / DB `input_json.views` 와 동일해야 함 */
export const VIEW_KEYS = ['front', 'top', 'left', 'right', 'back'] as const;
export type ViewKey = (typeof VIEW_KEYS)[number];
