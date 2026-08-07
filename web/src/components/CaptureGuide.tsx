/**
 * 촬영 위치 가이드.
 *
 * 파이프라인이 **카드 근처**에서만 물체를 찾는다(worker `height_segment` 의 ROI,
 * `_keep_near_card`). 배경이 복잡한 책상에서 잡동사니를 걸러내려면 필요한 제약인데,
 * 그걸 코드로만 요구하고 화면에서 안 알려주면 사용자는 계속 실패한다.
 *
 * 실측으로 확인된 실패 원인을 그대로 항목화했다.
 * - 물체가 카드에서 멀다 → ROI 밖이라 못 찾음
 * - 물체가 너무 작다 → 픽셀이 부족해 깊이가 뭉개짐
 * - 투명 케이스 안 → 케이스를 잼(실측: 케이스 모서리를 잡아 48g)
 */

type Props = { compact?: boolean };

const CARD = { x: 168, y: 60, w: 104, h: 66 };
// 카드 긴 변 1배 = worker ROI 반경과 같은 정의
const ROI_R = CARD.w;
const CARD_CX = CARD.x + CARD.w / 2;
const CARD_CY = CARD.y + CARD.h / 2;

export function CaptureGuide({ compact = false }: Props) {
  return (
    <div className="notice notice--tips" style={{ marginBottom: '1rem' }}>
      <p className="notice__title">어디에 놓고 찍나요</p>

      <svg
        viewBox="0 0 320 190"
        role="img"
        aria-label="신용카드 옆에 귀금속을 두고 촬영하는 위치 안내"
        style={{ width: '100%', maxWidth: 360, display: 'block', margin: '0.5rem auto' }}
      >
        {/* 프레임 */}
        <rect
          x="6" y="6" width="308" height="178" rx="8"
          fill="none" stroke="currentColor" strokeOpacity="0.25" strokeWidth="2"
        />
        <text x="16" y="24" fontSize="11" fill="currentColor" fillOpacity="0.5">
          사진 프레임
        </text>

        {/* 권장 영역 — worker ROI(카드 긴 변 1배, **카드 왼쪽 절반**)와 같은 정의 */}
        <path
          d={`M ${CARD_CX} ${CARD_CY - ROI_R} A ${ROI_R} ${ROI_R} 0 0 0 ${CARD_CX} ${CARD_CY + ROI_R} Z`}
          fill="#22c55e" fillOpacity="0.12"
          stroke="#22c55e" strokeOpacity="0.6" strokeWidth="1.5" strokeDasharray="5 4"
        />

        {/* 카드 */}
        <rect
          x={CARD.x} y={CARD.y} width={CARD.w} height={CARD.h} rx="6"
          fill="#3b82f6" fillOpacity="0.22" stroke="#3b82f6" strokeWidth="2"
        />
        <text
          x={CARD_CX} y={CARD_CY + 4}
          fontSize="11" textAnchor="middle" fill="#1d4ed8" fontWeight="600"
        >
          신용카드
        </text>

        {/* 물체 — 카드 왼쪽, ROI 안 */}
        <circle cx="120" cy="93" r="17" fill="#eab308" fillOpacity="0.30" stroke="#a16207" strokeWidth="2" />
        <text x="120" y="97" fontSize="10" textAnchor="middle" fill="#854d0e" fontWeight="600">
          귀금속
        </text>

        {/* 간격 */}
        <line x1="137" y1="93" x2={CARD.x} y2="93" stroke="#a16207" strokeWidth="1.5" strokeDasharray="3 3" />
        <text x="152" y="88" fontSize="9" textAnchor="middle" fill="#854d0e">
          바짝
        </text>

        <text x={CARD_CX} y={CARD_CY + ROI_R + 16} fontSize="10" textAnchor="middle" fill="#15803d">
          귀금속은 카드 왼쪽, 이 반원 안에
        </text>
      </svg>

      <ul style={{ margin: '0.25rem 0 0', paddingLeft: '1.15rem' }}>
        <li>
          <strong>귀금속은 왼쪽, 카드는 오른쪽</strong>에 두세요. 이 배치를 지키면 탐색 영역이
          절반으로 줄어 훨씬 정확하게 찾습니다.
        </li>
        <li>
          귀금속을 <strong>카드 바로 옆</strong>에 붙여 두세요. 카드에서 멀면 찾지 못합니다.
        </li>
        <li>
          <strong>카드와 같은 바닥 면</strong>에. 카드 위에 올리지 마세요.
        </li>
        <li>
          <strong>케이스·포장에서 꺼내</strong> 주세요. 투명 케이스도 그대로 재어 버립니다.
        </li>
        {!compact && (
          <>
            <li>
              둘 다 화면에 크게 나오게 <strong>가까이</strong>(약 25~35cm). 귀금속이 카드의{' '}
              <strong>1/3 이상</strong> 크기로 보이면 좋습니다.
            </li>
            <li>
              카드가 살짝 <strong>비스듬히</strong> 보이게 찍으면 거리 추정이 정확해집니다.
            </li>
            <li>
              책상 위 다른 물건은 되도록 치우고, <strong>무늬 없는 바닥</strong>에서 찍어 주세요.
            </li>
            <li>
              폰 <strong>기본 카메라</strong>로 찍어 그대로 올려 주세요. 메신저를 거치면 촬영 정보가
              지워져 정확도가 떨어집니다.
            </li>
          </>
        )}
      </ul>
    </div>
  );
}
