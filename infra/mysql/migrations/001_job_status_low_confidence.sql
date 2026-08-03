-- jobs.status 에 completed_low_confidence 추가.
--
-- 왜 필요한가: worker 가 soft 에러(재촬영으로 회복 가능)를 이 상태로 저장하는데,
-- infra/mysql/init/01-schema.sql 은 **볼륨이 비어 있을 때만** 실행된다.
-- 이미 만들어진 DB 는 ENUM 에 값이 없어 UPDATE 가 실패한다.
--
-- 멱등: 이미 값이 있으면 ALTER 를 건너뛴다.
SET @needs_alter := (
  SELECT COUNT(*) FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME = 'jobs'
    AND COLUMN_NAME = 'status'
    AND COLUMN_TYPE NOT LIKE '%completed_low_confidence%'
);

SET @stmt := IF(@needs_alter = 1,
  "ALTER TABLE jobs MODIFY COLUMN status ENUM('pending','processing','completed','completed_low_confidence','failed') NOT NULL DEFAULT 'pending'",
  'SELECT ''jobs.status already has completed_low_confidence'' AS note'
);
PREPARE s FROM @stmt;
EXECUTE s;
DEALLOCATE PREPARE s;
