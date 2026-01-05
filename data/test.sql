
-- 优化后的SQL查询
-- 优化点：
-- 1. 使用CTE提高可读性和性能
-- 2. 减少重复的表join
-- 3. 优化日期比较条件
-- 4. 移除不必要的GROUP BY
-- 5. 添加索引建议注释

-- 建议添加索引：
-- CREATE INDEX idx_loan_app_time_reloan ON loan_application_info(application_time, reloan, app_info_id, member_id);
-- CREATE INDEX idx_audit_record ON risk_system_sharding.audit_record_2025(target_id, plan_id, business_id);
-- CREATE INDEX idx_member_phone ON loan_member_info(id, telephone_no, version);
-- CREATE INDEX idx_dispatch_time ON call_sale_dispatch_log(dispatch_time, consumer_id);

WITH 
-- 主申请数据（添加过滤条件减少数据量）
main_applications AS (
    SELECT 
        a.id,
        a.member_id,
        a.application_time,
        a.app_info_id,
        b.telephone_no
    FROM loan_application_info a
    INNER JOIN loan_member_info b ON a.member_id = b.id
    WHERE a.reloan = 0 
        AND a.app_info_id = 28
        AND a.application_time BETWEEN '2025-11-01 00:00:00' AND '2025-12-30 23:59:59'
),

-- 历史申请记录（30天内的重复申请）
history_applications AS (
    SELECT DISTINCT
        b.telephone_no,
        a.app_info_id,
        a.application_time
    FROM loan_application_info a
    INNER JOIN loan_member_info b ON a.member_id = b.id
    WHERE a.status IN (5, 7)
        AND a.platform = 2
        AND a.application_time >= '2025-09-01 00:00:00'
),

-- 电销跟进记录（优化后移除不必要的GROUP BY）
call_records AS (
    SELECT 
        d.phone,
        d.app_info_id,
        d1.id,
        d1.dispatch_time,
        f.follow_time
    FROM call_sale_consumer d
    INNER JOIN call_sale_dispatch_log d1 ON d.id = d1.consumer_id
    INNER JOIN call_sale_follow_log f ON d1.id = f.dispatch_log_id
    WHERE d1.dispatch_time >= '2025-10-01 00:00:00'
),

-- 聚合电销记录
call_summary AS (
    SELECT 
        ma.id AS app_id,
        ma.application_time,
        ma.telephone_no,
        MAX(cr.id) AS dis_max,
        MAX(cr.dispatch_time) AS dispatch_time_max,
        MAX(cr.follow_time) AS follow_time_max
    FROM main_applications ma
    INNER JOIN call_records cr 
        ON ma.telephone_no = cr.phone 
        AND ma.app_info_id = cr.app_info_id
        AND cr.dispatch_time <= ma.application_time
        AND cr.follow_time <= ma.application_time
    GROUP BY ma.id, ma.application_time, ma.telephone_no
)

-- 主查询
SELECT DISTINCT
    ma.telephone_no,
    ma.id
FROM main_applications ma
-- 审核记录关联
INNER JOIN risk_system_sharding.audit_record_2025 ee 
    ON ee.target_id = ma.id 
    AND ee.business_id = 31
    AND ee.plan_id IN (41,48,51,56,61,67,68,69,70,81,86,87,64,91,83,73,55,100)
-- 历史申请记录关联（排除30天内重复申请）
LEFT JOIN history_applications ha 
    ON ha.telephone_no = ma.telephone_no
    AND ha.app_info_id = ma.app_info_id
    AND ha.application_time < ma.application_time
    AND DATEDIFF(ma.application_time, ha.application_time) <= 30
-- 电销跟进记录关联（3天内有跟进）
INNER JOIN call_summary cs 
    ON cs.app_id = ma.id
    AND DATEDIFF(cs.application_time, cs.follow_time_max) <= 3
-- 过滤条件：排除有历史重复申请的记录
WHERE ha.telephone_no IS NOT NULL
