import { Controller, Get } from '@nestjs/common';

/** Phase 6.3 — 초안 텍스트 (운영 전 법무 검토 필수, concept §17) */
@Controller('legal')
export class LegalController {
  @Get('terms')
  terms() {
    return {
      title: '이용약관 (초안)',
      body:
        '본 서비스는 참고 견적·정보 제공 목적이며 감정·법적 효력이 없습니다. ' +
        '텅 빈 링·속 빈 디자인 등 사진만으로 금속량을 특정할 수 없는 형태는 실제 질량과 추정치 차이가 클 수 있으며, 정확한 질량을 보장하지 않습니다. ' +
        '업로드 이미지는 운영 정책에 따라 보관 기간 후 삭제될 수 있습니다. (§17)',
    };
  }

  @Get('privacy')
  privacy() {
    return {
      title: '개인정보 처리방침 (초안)',
      body:
        '촬영·업로드 이미지 및 작업 메타데이터는 분석·품질 개선·법적 의무에 필요한 범위에서만 처리합니다. ' +
        '삭제 요청 절차는 고객센터 안내를 따릅니다. (§17)',
    };
  }
}
