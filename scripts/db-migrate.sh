#!/usr/bin/env bash
# infra/mysql/migrations/*.sql 을 번호순으로 적용한다.
#
# init/01-schema.sql 은 **빈 볼륨에서만** 실행되므로, 이미 만들어진 DB 에는
# 스키마 변경이 반영되지 않는다. 그 간극을 메우는 스크립트.
# 각 마이그레이션은 **멱등**하게 작성한다(여러 번 돌려도 안전).
set -euo pipefail
cd "$(dirname "$0")/.."

DB="${MYSQL_DATABASE:-archimedes}"
USER="${MYSQL_ROOT_USER:-root}"
PASS="${MYSQL_ROOT_PASSWORD:-rootsecret}"

shopt -s nullglob
FILES=(infra/mysql/migrations/*.sql)
if [ ${#FILES[@]} -eq 0 ]; then
  echo "적용할 마이그레이션이 없습니다."
  exit 0
fi

for f in "${FILES[@]}"; do
  echo "==> $f"
  docker compose exec -T mysql mysql -u"$USER" -p"$PASS" "$DB" < "$f"
done
echo "==> done."
