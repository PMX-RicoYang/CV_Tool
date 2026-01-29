/* eslint-disable no-console */
require('dotenv').config();
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { parse: csvParse } = require('csv-parse/sync');
const iconv = require('iconv-lite');
const dayjs = require('dayjs');
const utc = require('dayjs/plugin/utc');
const customParseFormat = require('dayjs/plugin/customParseFormat');
const { MongoClient, ObjectId  } = require('mongodb');
const cookieParser = require('cookie-parser');
const jwt = require('jsonwebtoken');
const crypto = require('crypto');


dayjs.extend(utc);
dayjs.extend(customParseFormat);

const app = express();
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true }));

// ---- ENV ----
const MONGO_URI = process.env.MONGO_URI || 'mongodb://rootAMR:AMR28006928@mongodb:27017/?authSource=admin';
const MONGO_DB = process.env.MONGO_DB || 'shop';
const MONGO_COL = process.env.MONGO_COL || 'quality';
const PORT = process.env.PORT ? Number(process.env.PORT) : 5001;

// ---- Mongo ----
const mongoClient = new MongoClient(MONGO_URI);
let col;
let devCol;
let settingCol;

(async () => {
  await mongoClient.connect();
  const db = mongoClient.db(MONGO_DB);

  col = db.collection(MONGO_COL);
  col_exception = db.collection('exception_quality');
  devCol = db.collection('devices');
  settingCol = db.collection('settings');

  await col.createIndex({ production_id: 1 }, { unique: true });
  await col_exception.createIndex({ production_id: 1 }, { unique: true });
  await devCol.createIndex({ device_id: 1 }, { unique: true, name: 'uniq_device_id' });
  // ✅ 這裡新增 → 為 quality 建複合索引（加速店鋪查詢 & 時間查詢）
  await col.createIndex({ device_id: 1, event_time: 1 }, { name: 'idx_device_time' });

  await col_exception.createIndex({ device_id: 1, event_time: 1 }, { name: 'idx_device_time' });

  console.log(`[mongo] connected → ${MONGO_URI}`);
})();

const DEFAULT_MAX_PRODUCTION_SEC = 120;
const DEFAULT_MIN_SERVING_TEMP   = 75;
const DEFAULT_MISSING_AS_NG      = true;

// 小工具：把任意輸入轉成 boolean
function parseBoolLoose(v, defaultVal = false) {
  if (v === undefined || v === null) return defaultVal;
  const s = String(v).trim().toLowerCase();
  if (['1', 'true', 'yes', 'y'].includes(s)) return true;
  if (['0', 'false', 'no', 'n'].includes(s)) return false;
  return defaultVal;
}


// ---- Time helpers ----
function parseISO(s) {
  if (!s) return null;
  let str = String(s).trim();
  if (!str) return null;
  // 支援末尾 Z 與無 Z（皆視為 UTC）
  if (str.endsWith('Z')) {
    const d = dayjs.utc(str);
    return d.isValid() ? d.toDate() : null;
  }
  // 無 Z：直接以 UTC 解析
  const d = dayjs.utc(str);
  return d.isValid() ? d.toDate() : null;
}

function parseDtStrict(s) {
  if (!s) return null;
  const str = String(s).trim();
  if (!str) return null;
  // ISO（2025-11-03T11:26:37.000Z）
  if (/^\d{4}-\d{2}-\d{2}T/.test(str)) {
    const d = dayjs(str);   // 讓 dayjs 自動 parse ISO
    if (d.isValid()) return d.toDate();
  }
  const fmts = [
    'YYYY-MM-DD HH:mm',
    'YYYY/MM/DD HH:mm',
    'YYYY/MM/DD HH:mm:ss',
    'YYYY/MM/DD HH:mm:ss.SSS',
    'YYYY-MM-DD HH:mm:ss'
  ];

  for (const f of fmts) {
    const d = dayjs.utc(str, f, true);
    if (d.isValid()) return d.toDate();
  }

  return null;
}

// ---- CSV helpers ----
function stripBOM(text) {
  if (!text) return text;
  return text.charCodeAt(0) === 0xFEFF ? text.slice(1) : text;
}
function safeFloat(v) {
  if (v === null || v === undefined) return null;
  const num = Number(String(v).trim());
  return Number.isFinite(num) ? num : null;
}

// ---- Upload ----
const upload = multer({ storage: multer.memoryStorage() });

// =====================================================================
// POST /devices
// Body(JSON):
//   {
//     "device_id": "DEV-0001",                // 必填
//     "created_at": "2025-11-03T09:30:00Z",   // 選填，不給就用現在(UTC)
//     "store": {                              // 必填
//       "name": "台北南京店",
//       "number": "S001",
//       "market": "北一區"
//     }
//   }
// 行為：以 device_id upsert；第一次新增、之後更新 store；維持 created_at 不變
// 回傳：{ ok:true, upserted:bool, device_id, doc }
// =====================================================================
app.post('/devices', async (req, res) => {
    if (!devCol) return res.status(503).json({ ok: false, error: 'devices collection not ready' });
  
    try {
      const { device_id, created_at, store } = req.body || {};
      // 基本驗證
      if (!device_id || typeof device_id !== 'string' || !device_id.trim()) {
        return res.status(400).json({ ok: false, error: 'device_id is required (non-empty string)' });
      }
      if (!store || typeof store !== 'object') {
        return res.status(400).json({ ok: false, error: 'store is required (object)' });
      }
      const { name, number, market } = store;
      if (!name || !number || !market) {
        return res.status(400).json({ ok: false, error: 'store.name, store.number, store.market are required' });
      }
  
      // 時間處理
      const now = new Date();
      const createdAt = created_at ? parseISO(created_at) : now;
      if (created_at && !createdAt) {
        return res.status(400).json({ ok: false, error: 'created_at must be ISO datetime (e.g. 2025-11-03T09:30:00Z)' });
      }
  
      // upsert：第一次插入時寫入 created_at；之後更新不覆蓋 created_at
      const r = await devCol.updateOne(
        { device_id: device_id.trim() },
        {
          $setOnInsert: {
            device_id: device_id.trim(),
            created_at: createdAt || now,
            created_date: dayjs.utc(createdAt || now).startOf('day').toDate()
          },
          $set: {
            store: {
              name: String(name).trim(),
              number: String(number).trim(),
              market: String(market).trim()
            },
            updated_at: now
          }
        },
        { upsert: true }
      );
  
      const upserted = !!r.upsertedId;
      const doc = await devCol.findOne({ device_id: device_id.trim() }, { projection: { _id: 0 } });
  
      return res.json({ ok: true, upserted, device_id: device_id.trim(), doc });
    } catch (err) {
      // 唯一鍵衝突等錯誤處理
      if (err && err.code === 11000) {
        return res.status(409).json({ ok: false, error: 'device_id already exists (unique constraint)', code: 11000 });
      }
      console.error('[POST /devices] error:', err);
      return res.status(500).json({ ok: false, error: String(err && err.message || err) });
    }
});

// ===================================================================
// POST /ingest/csv
// form-data:
//   files[]: 多檔 CSV
//   force_utf8: "1"/"true" 時強制用 utf-8-sig/utf-8 嘗試解碼
// 功能新增：
//   - 解析「Serving Time - Back」「Soup Time」「Serving Temp - Back」「拉麵品名」
//   - 計算 production_time_sec（Serving Time - Back 與 Soup Time 的秒差）
//   - 取出 serving_temp（Serving Temp - Back）
//   - ingredient_missing：拉麵品名 == "Unknown"
//   - status_norm：任一成立（prod_time>60 / temp<75 / ingredients 缺）則 'ng'，否則 'ok'
//   - 若無 Start/End Time，則 start_time=Soup Time、end_time=Serving Time - Back、event_time=end_time
// ===================================================================
app.post('/ingest/csv', upload.array('files'), async (req, res) => {
  if (!req.files || !req.files.length) {
    return res.status(400).json({
      ok: false,
      inserted: 0,
      insertedException: 0,
      errors: ['no files[] uploaded'],
    });
  }

  const forceUtf8 = String(req.body.force_utf8 || '0').toLowerCase();
  const enforceUtf8 = forceUtf8 === '1' || forceUtf8 === 'true';

  let inserted = 0;
  let insertedException = 0;
  const errors = [];

  // === 欄位名稱集中定義 ===
  const FIELD = {
    // 舊欄位（目前只留 End / Final Temp）
    END_TIME: 'End Time',
    FINAL_TEMP: 'Final Temp',

    // 新欄位
    SERVING_TIME_BACK: 'Serving Time - Back',
    SOUP_TIME: 'Soup Time',
    EMPTY_BOWL_TIME: 'Empty Bowl Time',
    SERVING_TEMP_BACK: 'Serving Temp - Back',
    RAMEN_NAME: 'ramen_name',
    DEVICE_ID: 'Store ID',

    // ✅ 新增：Area（CSV 可能有多個同名欄位）
    AREA: 'Area',
  };

  // ✅ 取出 row 裡所有 Area 欄位（支援 Area、Area_1、Area_2...）
  function getAllAreaValues(row) {
    const vals = [];
    for (const [k, v] of Object.entries(row || {})) {
      if (typeof k !== 'string') continue;
      // columns: true 時，重名欄位常見會變成 Area, Area_1, Area_2...
      if (k === FIELD.AREA || k.startsWith(FIELD.AREA + '_')) {
        const s = String(v ?? '').trim();
        if (s) vals.push(s);
      }
    }
    return vals;
  }

  // ✅ 判斷 Area 規則：
  // - 只要任何一個 Area 欄位是 outside => 直接視為 exception（強制）
  // - 否則若至少一個 Area 欄位是 inside => 允許進正常資料
  // - 若沒有任何 Area（或值不在 inside/outside）=> 視為 unknown，這裡我採「不擋」但可自行改成 exception
  function areaDecision(row) {
    const areas = getAllAreaValues(row).map(s => s.toLowerCase());
    const hasOutside = areas.includes('outside');
    const hasInside = areas.includes('inside');

    if (hasOutside) return { okToIngest: false, toException: true, reason: 'area_outside' };
    if (hasInside) return { okToIngest: true,  toException: false, reason: 'area_inside' };

    // 沒有 Area 或不認得的值：不當 exception（你若要更嚴格可改成 toException: true）
    return { okToIngest: true, toException: false, reason: 'area_unknown_or_missing' };
  }

  for (const f of req.files) {
    try {
      // ===== 1) 解碼 CSV 內容 =====
      const raw = f.buffer;
      let content = null;

      const encodings = enforceUtf8
        ? ['utf-8-sig', 'utf-8']
        : ['utf-8-sig', 'utf-8'];

      for (const enc of encodings) {
        try {
          content = iconv.decode(raw, enc);
          if (content) break;
        } catch (_) {
          // 忽略單一編碼失敗，嘗試下一個
        }
      }
      if (!content) {
        throw new Error('decode failed');
      }

      content = stripBOM(content);

      const records = csvParse(content, {
        columns: true,
        skip_empty_lines: true,
        trim: true,
      });

      if (!records.length) {
        // 空檔案就跳過，不視為錯誤
        continue;
      }

      const batch = [];
      const batchException = [];

      // ===== 2) 每筆 row 轉成 doc =====
      for (const row of records) {
        // ✅ Area gate：outside => exception；inside => 繼續；其他 => 繼續（依上面策略）
        const area = areaDecision(row);
        if (!area.okToIngest && area.toException) {
          // outside：直接丟 exception（也可選擇不寫入 raw，但你說要 exception，所以照做）
          // 但還是要組 production_id，需要 event_time；若連時間都沒有，就略過（沒 key 無法存）
          const servingTime    = parseDtStrict(row[FIELD.SERVING_TIME_BACK]);
          const soupTime       = parseDtStrict(row[FIELD.SOUP_TIME]);
          const emptyBowlTime  = parseDtStrict(row[FIELD.EMPTY_BOWL_TIME]);
          const event_time = emptyBowlTime || servingTime || soupTime || null;
          if (!event_time) {
            console.log('[/ingest/csv] outside row 缺少 event_time，略過');
            continue;
          }

          const device_id = row[FIELD.DEVICE_ID] ? String(row[FIELD.DEVICE_ID]).trim() : null;
          const production_id = device_id && event_time
            ? `${device_id}_${event_time.getTime()}`
            : `unknown_${event_time.getTime()}`;

          const docEx = {
            production_id,
            raw: row,
            device_id,
            start_time: soupTime || null,
            end_time: servingTime || null,
            event_time,
            final_temp: safeFloat(row[FIELD.FINAL_TEMP]),
            serving_temp: safeFloat(row[FIELD.SERVING_TEMP_BACK]),
            production_time_sec: (() => {
              if (servingTime && soupTime && servingTime >= soupTime) return (servingTime - soupTime) / 1000.0;
              return 0;
            })(),
            ingredient_missing: (() => {
              const ramenRaw = String(row[FIELD.RAMEN_NAME] || '').trim();
              const ramenLc  = ramenRaw.toLowerCase();
              return ramenLc === 'unknown' || ramenRaw === '6';
            })(),
            status_norm: 'exception',
            exception_reason: area.reason, // ✅ 記錄是 area_outside
          };

          docEx.event_date = dayjs.utc(event_time).startOf('day').toDate();
          batchException.push(docEx);
          continue;
        }

        // ---- 解析時間 ----
        const servingTime    = parseDtStrict(row[FIELD.SERVING_TIME_BACK]); // 端出時間(Back)
        const soupTime       = parseDtStrict(row[FIELD.SOUP_TIME]);        // 湯時間
        const emptyBowlTime  = parseDtStrict(row[FIELD.EMPTY_BOWL_TIME]);  // 空碗時間

        // 決定主時間 event_time（優先 emptyBowl，其次 serving，再次 soup）
        const event_time = emptyBowlTime || servingTime || soupTime || null;
        if (!event_time) {
          // 缺少關鍵時間，跳過該 row
          console.log('[/ingest/csv] 缺少關鍵欄位 event_time');
          continue;
        }

        // 定義 start/end，例如湯開始 → 端出
        const start_time = soupTime || null;
        const end_time   = servingTime || null;

        // ---- 解析溫度 ----
        const servingTemp = safeFloat(row[FIELD.SERVING_TEMP_BACK]);
        const finalTemp   = safeFloat(row[FIELD.FINAL_TEMP]); // 保留 Final Temp

        // ---- 生產時間（秒） ----
        let productionSec = 0;
        if (servingTime && soupTime && servingTime >= soupTime) {
          productionSec = (servingTime - soupTime) / 1000.0;
        }

        // ---- 食材缺失判斷 ----
        const ramenRaw = String(row[FIELD.RAMEN_NAME] || '').trim();
        const ramenLc  = ramenRaw.toLowerCase();

        const ingredientMissing =
          ramenLc === 'unknown' ||
          ramenRaw === '6';

        // ---- NG 判斷 ----
        const isNg =
          (productionSec !== null && productionSec > 120) ||
          (servingTemp !== null && servingTemp < 75) ||
          ingredientMissing === true;

        const statusNorm = isNg ? 'ng' : 'ok';

        // ---- 裝置 / production_id ----
        const device_id = row[FIELD.DEVICE_ID]
          ? String(row[FIELD.DEVICE_ID]).trim()
          : null;

        const production_id = device_id && event_time
          ? `${device_id}_${event_time.getTime()}`
          : `unknown_${event_time.getTime()}`;

        const doc = {
          production_id,
          raw: row,
          device_id,
          start_time,
          end_time,
          event_time,
          final_temp: finalTemp,
          serving_temp: servingTemp,
          production_time_sec: productionSec,
          ingredient_missing: ingredientMissing,
          status_norm: statusNorm,

          // ✅ 可選：把 Area 決策也記一下，方便查核
          area_decision: area.reason,
        };

        // event_date（該日 00:00Z）
        doc.event_date = dayjs.utc(event_time).startOf('day').toDate();

        // ---- 例外 vs 正常 批次 ----
        const deviceLc = (doc.device_id ? String(doc.device_id).toLowerCase() : '');

        const isException =
          ramenLc === 'exception' ||
          ramenRaw === '7' ||
          ramenRaw === '12' ||
          deviceLc === 'test';

        if (isException) {
          batchException.push(doc);
        } else {
          batch.push(doc);
        }
      } // end for each row

      // ===== 3) 寫入 Mongo：正常批次 =====
      if (batch.length) {
        const ops = batch.map(doc => ({
          updateOne: {
            filter: { production_id: doc.production_id },
            update: { $set: doc },
            upsert: true,
          },
        }));

        const r = await col.bulkWrite(ops, { ordered: false });
        inserted += r.upsertedCount || 0;
      }

      // ===== 4) 寫入 Mongo：例外批次 =====
      if (batchException.length) {
        const opsEx = batchException.map(doc => ({
          updateOne: {
            filter: { production_id: doc.production_id },
            update: { $set: doc },
            upsert: true,
          },
        }));

        const rex = await col_exception.bulkWrite(opsEx, { ordered: false });
        insertedException += rex.upsertedCount || 0;
      }
    } catch (e) {
      errors.push(`${f.originalname || 'file'}: ${e.message || String(e)}`);
    }
  }

  return res.json({
    ok: true,
    inserted,
    insertedException,
    errors,
  });
});

// app.post('/ingest/csv', upload.array('files'), async (req, res) => {
//   if (!req.files || !req.files.length) {
//     return res.status(400).json({
//       ok: false,
//       inserted: 0,
//       insertedException: 0,
//       errors: ['no files[] uploaded'],
//     });
//   }

//   const forceUtf8 = String(req.body.force_utf8 || '0').toLowerCase();
//   const enforceUtf8 = forceUtf8 === '1' || forceUtf8 === 'true';

//   let inserted = 0;
//   let insertedException = 0;
//   const errors = [];

//   // === 欄位名稱集中定義 ===
//   const FIELD = {
//     // 舊欄位（目前只留 End / Final Temp）
//     END_TIME: 'End Time',
//     FINAL_TEMP: 'Final Temp',

//     // 新欄位
//     SERVING_TIME_BACK: 'Serving Time - Back',
//     SOUP_TIME: 'Soup Time',
//     EMPTY_BOWL_TIME: 'Empty Bowl Time',
//     SERVING_TEMP_BACK: 'Serving Temp - Back',
//     RAMEN_NAME: 'ramen_name',
//     DEVICE_ID: 'Store ID',
//   };

//   for (const f of req.files) {
//     try {
//       // ===== 1) 解碼 CSV 內容 =====
//       const raw = f.buffer;
//       let content = null;

//       const encodings = enforceUtf8
//         ? ['utf-8-sig', 'utf-8']
//         : ['utf-8-sig', 'utf-8'];

//       for (const enc of encodings) {
//         try {
//           content = iconv.decode(raw, enc);
//           if (content) break;
//         } catch (_) {
//           // 忽略單一編碼失敗，嘗試下一個
//         }
//       }
//       if (!content) {
//         throw new Error('decode failed');
//       }

//       content = stripBOM(content);

//       const records = csvParse(content, {
//         columns: true,
//         skip_empty_lines: true,
//         trim: true,
//       });

//       if (!records.length) {
//         // 空檔案就跳過，不視為錯誤
//         continue;
//       }

//       const batch = [];
//       const batchException = [];

//       // ===== 2) 每筆 row 轉成 doc =====
//       for (const row of records) {
//         // ---- 解析時間 ----
//         const servingTime    = parseDtStrict(row[FIELD.SERVING_TIME_BACK]); // 端出時間(Back)
//         const soupTime       = parseDtStrict(row[FIELD.SOUP_TIME]);        // 湯時間
//         const emptyBowlTime  = parseDtStrict(row[FIELD.EMPTY_BOWL_TIME]);  // 空碗時間

//         // 決定主時間 event_time（優先 emptyBowl，其次 serving，再次 soup）
//         const event_time = emptyBowlTime || servingTime || soupTime || null;
//         if (!event_time) {
//           // 缺少關鍵時間，跳過該 row
//           console.log('[/ingest/csv] 缺少關鍵欄位 event_time');
//           continue;
//         }

//         // 定義 start/end，例如湯開始 → 端出
//         const start_time = soupTime || null;
//         const end_time   = servingTime || null;

//         // ---- 解析溫度 ----
//         const servingTemp = safeFloat(row[FIELD.SERVING_TEMP_BACK]);
//         const finalTemp   = safeFloat(row[FIELD.FINAL_TEMP]); // 保留 Final Temp

//         // ---- 生產時間（秒） ----
//         let productionSec = 0;
//         if (servingTime && soupTime && servingTime >= soupTime) {
//           productionSec = (servingTime - soupTime) / 1000.0;
//         }

//         // ---- 食材缺失判斷 ----
//         const ramenRaw = String(row[FIELD.RAMEN_NAME] || '').trim();
//         const ramenLc  = ramenRaw.toLowerCase();

//         const ingredientMissing =
//           ramenLc === 'unknown' ||
//           ramenRaw === '6';

//         // ---- NG 判斷 ----
//         const isNg =
//           (productionSec !== null && productionSec > 120) ||
//           (servingTemp !== null && servingTemp < 75) ||
//           ingredientMissing === true;

//         const statusNorm = isNg ? 'ng' : 'ok';

//         // ---- 裝置 / production_id ----
//         const device_id = row[FIELD.DEVICE_ID]
//           ? String(row[FIELD.DEVICE_ID]).trim()
//           : null;

//         const production_id = device_id && event_time
//           ? `${device_id}_${event_time.getTime()}`
//           : `unknown_${event_time.getTime()}`;

//         const doc = {
//           production_id,
//           raw: row,
//           device_id,
//           start_time,
//           end_time,
//           event_time,
//           final_temp: finalTemp,
//           serving_temp: servingTemp,
//           production_time_sec: productionSec,
//           ingredient_missing: ingredientMissing,
//           status_norm: statusNorm,
//         };

//         // event_date（該日 00:00Z）
//         doc.event_date = dayjs.utc(event_time).startOf('day').toDate();

//         // ---- 例外 vs 正常 批次 ----
//         const isException =
//           ramenLc === 'exception' || ramenRaw === '7' || doc.device_id.toLowerCase() === 'test';

//         if (isException) {
//           batchException.push(doc);
//         } else {
//           batch.push(doc);
//         }
//       } // end for each row

//       // ===== 3) 寫入 Mongo：正常批次 =====
//       if (batch.length) {
//         const ops = batch.map(doc => ({
//           updateOne: {
//             filter: { production_id: doc.production_id },
//             update: { $set: doc },
//             upsert: true,
//           },
//         }));

//         const r = await col.bulkWrite(ops, { ordered: false });
//         inserted += r.upsertedCount || 0;
//       }

//       // ===== 4) 寫入 Mongo：例外批次 =====
//       if (batchException.length) {
//         const opsEx = batchException.map(doc => ({
//           updateOne: {
//             filter: { production_id: doc.production_id },
//             update: { $set: doc },
//             upsert: true,
//           },
//         }));

//         const rex = await col_exception.bulkWrite(opsEx, { ordered: false });
//         insertedException += rex.upsertedCount || 0;
//       }
//     } catch (e) {
//       errors.push(`${f.originalname || 'file'}: ${e.message || String(e)}`);
//     }
//   }

//   return res.json({
//     ok: true,
//     inserted,
//     insertedException,
//     errors,
//   });
// });


// ============================================================
// GET /quality/entries
// 參數：
//   from, to          : ISO 時間（event_time 篩選）
//   include_raw       : 1/true 時附 raw
//   include_unknown   : 1/true 時把沒有對應店鋪（無 devices 紀錄）的也納入，store=UNKNOWN
//   market            : 依市場名過濾（精確比對）
//   store_number      : 依店號過濾（精確比對）
//   store_name        : 依店名過濾（精確比對）
//   device_id         : 指定裝置（可逗號分隔多個）
// 回傳：[{ time, final_temp, duration_sec, roi_id?, store:{name,number,market}, raw? }]
// ============================================================
app.get('/quality/entries', async (req, res) => {
  try {
    const includeRaw = ['1','true','True'].includes(String(req.query.include_raw || '0'));
    const includeUnknown = ['1','true','True'].includes(String(req.query.include_unknown || '0'));

    const tFrom = parseISO(req.query.from);
    const tTo   = parseISO(req.query.to);

    const market = req.query.market ? String(req.query.market).trim() : null;
    const storeNumber = req.query.store_number ? String(req.query.store_number).trim() : null;
    const storeName = req.query.store_name ? String(req.query.store_name).trim() : null;

    let deviceIds = null;
    if (req.query.device_id) {
      deviceIds = String(req.query.device_id)
        .split(',').map(s => s.trim()).filter(Boolean);
      if (!deviceIds.length) deviceIds = null;
    }

    const match = {};
    if (tFrom || tTo) {
      match.event_time = {};
      if (tFrom) match.event_time.$gte = tFrom;
      if (tTo)   match.event_time.$lte = tTo;
    }
    if (deviceIds) {
      match.device_id = { $in: deviceIds };
    }

    const pipe = [];
    if (Object.keys(match).length) pipe.push({ $match: match });

    // 連 devices 取得店鋪資訊
    pipe.push(
      { $lookup: {
          from: 'devices',
          localField: 'device_id',
          foreignField: 'device_id',
          as: 'dv'
        }
      },
      { $unwind: { path: '$dv', preserveNullAndEmptyArrays: includeUnknown } },
    );

    // 店鋪過濾（若未開 include_unknown，先排除沒有店鋪的）
    const storeFilter = {};
    if (!includeUnknown) {
      storeFilter['dv.store.name'] = { $ne: null };
      storeFilter['dv.store.number'] = { $ne: null };
      storeFilter['dv.store.market'] = { $ne: null };
    }
    if (market)      storeFilter['dv.store.market'] = market;
    if (storeNumber) storeFilter['dv.store.number'] = storeNumber;
    if (storeName)   storeFilter['dv.store.name']   = storeName;
    if (Object.keys(storeFilter).length) pipe.push({ $match: storeFilter });

    // 計算 durationSec，並套用原過濾條件
    pipe.push(
      { $match: {
          production_time_sec: { $ne: null },
          // durationSec: { $ne: null},
          serving_temp: { $ne: null }
          // durationSec: { $ne: null, $lte: 100 },
          // final_temp: { $ne: null, $gte: 50 }
        }
      },
      { $sort: { event_time: 1 } },
      { $project: Object.assign({
          _id: 1,
          time: { $dateToString: { format: '%Y-%m-%dT%H:%M:%SZ', date: '$event_time', timezone: 'UTC' } },
          serving_temp: 1,
          production_time_sec: 1,
          roi_id: 1,
          store: {
            name:   includeUnknown ? { $ifNull: ['$dv.store.name', 'UNKNOWN'] }   : '$dv.store.name',
            number: includeUnknown ? { $ifNull: ['$dv.store.number', 'UNKNOWN'] } : '$dv.store.number',
            market: includeUnknown ? { $ifNull: ['$dv.store.market', 'UNKNOWN'] } : '$dv.store.market'
          }
        }, includeRaw ? { raw: '$raw' } : {})
      }
    );

    const rows = await col.aggregate(pipe).toArray();
    return res.json({ data: rows });
  } catch (err) {
    console.error('[GET /quality/entries] error:', err);
    return res.status(500).json({ ok: false, error: String(err && err.message || err) });
  }
});

// =====================================================================
// GET /quality/temp-time-stats?from=...&to=...&temp_threshold=75&dur_threshold_sec=60&tz=UTC
// =====================================================================
app.get('/quality/temp-time-stats', async (req, res) => {
  const tempThreshold = Number(req.query.temp_threshold || 75);
  const durThresholdSec = Number(req.query.dur_threshold_sec || 120);
  const tFrom = parseISO(req.query.from);
  const tTo = parseISO(req.query.to);
  const tz = String(req.query.tz || 'UTC');

  console.log(`[/quality/temp-time-stats] tFrom = ${tFrom}  tTo = ${tTo}`);

  const match = {};
  if (tFrom || tTo) {
    match.event_time = {};
    if (tFrom) match.event_time.$gte = tFrom;
    if (tTo) match.event_time.$lte = tTo;
  }

  const pipe = [];
  if (Object.keys(match).length) pipe.push({ $match: match });

  pipe.push(
    {
      $addFields: {
        durationMs: {
          $cond: [
            { $and: [{ $ne: ['$start_time', null] }, { $ne: ['$end_time', null] }, { $gte: ['$end_time', '$start_time'] }] },
            { $subtract: ['$end_time', '$start_time'] },
            null
          ]
        }
      }
    },
    {
      $addFields: {
        durationSec: {
          $cond: [{ $ne: ['$durationMs', null] }, { $divide: ['$durationMs', 1000] }, null]
        }
      }
    },
    {
      $addFields: {
        temp_flag: {
          $cond: [
            { $and: [{ $ne: ['$final_temp', null] }, { $lt: ['$final_temp', tempThreshold] }] },
            true, false
          ]
        },
        known_temp: { $ne: ['$final_temp', null] },
        long_flag: {
          $cond: [
            { $and: [{ $ne: ['$durationSec', null] }, { $gt: ['$durationSec', durThresholdSec] }] },
            true, false
          ]
        },
        known_long: { $ne: ['$durationSec', null] },
        dateStr: { $dateToString: { format: '%Y-%m-%d', date: '$event_time', timezone: tz } }
      }
    },
    {
      $project: {
        dateStr: 1,
        only_low: { $cond: [{ $and: ['$known_temp', '$known_long', '$temp_flag', { $not: ['$long_flag'] }] }, 1, 0] },
        only_long: { $cond: [{ $and: ['$known_temp', '$known_long', { $not: ['$temp_flag'] }, '$long_flag'] }, 1, 0] },
        both: { $cond: [{ $and: ['$known_temp', '$known_long', '$temp_flag', '$long_flag'] }, 1, 0] },
        normal: { $cond: [{ $and: ['$known_temp', '$known_long', { $not: ['$temp_flag'] }, { $not: ['$long_flag'] }] }, 1, 0] },
        skipped: { $cond: [{ $or: [{ $not: ['$known_temp'] }, { $not: ['$known_long'] }] }, 1, 0] }
      }
    },
    {
      $group: {
        _id: '$dateStr',
        only_low: { $sum: '$only_low' },
        only_long: { $sum: '$only_long' },
        both: { $sum: '$both' },
        normal: { $sum: '$normal' },
        skipped: { $sum: '$skipped' }
      }
    },
    { $sort: { _id: 1 } },
    {
      $project: {
        _id: 0,
        date: '$_id',
        date_iso: { $concat: ['$_id', 'T00:00:00Z'] },
        only_low_count: '$only_low',
        only_long_count: '$only_long',
        both_count: '$both',
        normal_count: '$normal',
        skipped: '$skipped',
        total: { $add: ['$only_low', '$only_long', '$both', '$normal'] }
      }
    }
  );

  const rows = await col.aggregate(pipe).toArray();

  const data = [];
  let sum_only_low = 0, sum_only_long = 0, sum_both = 0, sum_normal = 0, sum_skipped = 0;
  for (const r of rows) {
    const only_low = Number(r.only_low_count || 0);
    const only_long = Number(r.only_long_count || 0);
    const both = Number(r.both_count || 0);
    const normal = Number(r.normal_count || 0);
    const skipped = Number(r.skipped || 0);

    sum_only_low += only_low;
    sum_only_long += only_long;
    sum_both += both;
    sum_normal += normal;
    sum_skipped += skipped;

    data.push({
      date: r.date,
      date_iso: r.date_iso,
      only_low_count: only_low,
      only_long_count: only_long,
      both_count: both,
      normal_count: normal,
      total: Number(r.total || 0),
      skipped
    });
  }

  return res.json({
    data,
    summary: {
      only_low_total: sum_only_low,
      only_long_total: sum_only_long,
      both_total: sum_both,
      normal_total: sum_normal,
      total: sum_only_low + sum_only_long + sum_both + sum_normal,
      skipped_total: sum_skipped,
      temp_threshold: tempThreshold,
      dur_threshold_sec: durThresholdSec,
      tz
    }
  });
});

// =====================================================================
// GET /quality/ok-ratio?from=...&to=...&status_field=狀態&ok_value=OK&tz=UTC
// =====================================================================
app.get('/quality/ok-ratio', async (req, res) => {
  const tFrom = parseISO(req.query.from);
  const tTo = parseISO(req.query.to);
  const statusField = String(req.query.status_field || '狀態');
  const okValueRaw = String(req.query.ok_value || 'OK');
  const okLc = okValueRaw.toLowerCase();
  const tz = String(req.query.tz || 'UTC');

  const match = {};
  if (tFrom || tTo) {
    match.event_time = {};
    if (tFrom) match.event_time.$gte = tFrom;
    if (tTo) match.event_time.$lte = tTo;
  }

  const statusKey = `raw.${statusField}`;

  const pipe = [];
  if (Object.keys(match).length) pipe.push({ $match: match });

  pipe.push(
    {
      $project: {
        dateStr: { $dateToString: { format: '%Y-%m-%d', date: '$event_time', timezone: tz } },
        status_norm: {
          $toLower: {
            $trim: {
              input: {
                $ifNull: [
                  '$status_norm',
                  { $ifNull: [`$${statusKey}`, { $ifNull: ['$raw.status', null] }] }
                ]
              }
            }
          }
        }
      }
    },
    {
      $group: {
        _id: '$dateStr',
        ok_count: { $sum: { $cond: [{ $eq: ['$status_norm', okLc] }, 1, 0] } },
        total: { $sum: 1 }
      }
    },
    { $sort: { _id: 1 } },
    {
      $project: {
        _id: 0,
        date: '$_id',
        date_iso: { $concat: ['$_id', 'T00:00:00Z'] },
        ok_count: 1,
        total: 1,
        ok_ratio: {
          $cond: [{ $gt: ['$total', 0] }, { $divide: ['$ok_count', '$total'] }, 0]
        }
      }
    }
  );

  const rows = await col.aggregate(pipe).toArray();

  const data = [];
  let okSum = 0, totalSum = 0;
  for (const r of rows) {
    const ok_c = Number(r.ok_count || 0);
    const tot = Number(r.total || 0);
    const ratio = Number(r.ok_ratio || 0);
    okSum += ok_c;
    totalSum += tot;
    data.push({
      date: r.date,
      ok_ratio: Number(ratio.toFixed(6)),
      ok_count: ok_c,
      total: tot,
      date_iso: r.date_iso
    });
  }

  const summary = {
    ok_total: okSum,
    total: totalSum,
    ok_ratio: totalSum > 0 ? okSum / totalSum : 0,
    status_field: statusField,
    ok_value: okValueRaw,
    tz
  };

  return res.json({ data, summary });
});

// =====================================================================
// GET /quality/store-ok-stats
// 參數：
//   from, to        : ISO 時間；用 event_time 篩選
//   ok_value        : 判定 OK 的文字（預設 "OK"，大小寫不敏感）
//   tz              : 例如 "Asia/Taipei"（預設 UTC，用於切日）
//   include_unknown : 1/true 時，沒有對應 devices.store 的紀錄也納入（店名=UNKNOWN）
// 輸出：
//   {
//     data: {
//       per_store: [{market, store_number, store_name, ok_count, total, ok_ratio}],
//       daily_store_category: [{
//         date, date_iso,
//         excellent_count, watch_count, bad_count,
//         total_stores, excellent_ratio, watch_ratio, bad_ratio
//       }]
//     },
//     summary: { tz, ok_value, thresholds: { excellent_gt:0.9, watch_ge:0.8, watch_lt:0.9 } }
//   }
// 說明：
//   - 店鋪來自 devices 集合（device_id 對應 devices.device_id；使用 devices.store.{name,number,market}）
//   - 評價分級：>90% 優秀(excellent)，80~89% 待觀察(watch)，<80% 不良(bad)
// =====================================================================
app.get('/quality/store-ok-stats', async (req, res) => {
  try {

    
    const tFrom = parseISO(req.query.from);
    const tTo   = parseISO(req.query.to);
    const okLc  = String(req.query.ok_value || 'OK').toLowerCase();
    const tz    = String(req.query.tz || 'UTC');
    const includeUnknown = ['1','true','True'].includes(String(req.query.include_unknown || '0'));

    // 動態判斷門檻（對 status_norm）
    const productionLimitSec = Number.isFinite(Number(req.query.max_production_sec))
      ? Number(req.query.max_production_sec)
      : 120;  // 預設：>60 秒為 NG

    const servingTempMin = Number.isFinite(Number(req.query.min_serving_temp))
      ? Number(req.query.min_serving_temp)
      : 75;  // 預設：<75 度為 NG

    // 食材缺失是否算 NG
    const missingAsNgRaw = String(req.query.missing_as_ng ?? '1').toLowerCase();
    const useMissingForNg = !['0', 'false', 'no'].includes(missingAsNgRaw);

    const match = {};
    if (tFrom || tTo) {
      match.event_time = {};
      if (tFrom) match.event_time.$gte = tFrom;
      if (tTo)   match.event_time.$lte = tTo;
    }

    // 狀態條件（用 JS 組好 expression）
    const ngBranches = [
      {
        $and: [
          { $ne: ['$production_time_sec', null] },
          { $gt: ['$production_time_sec', productionLimitSec] }
        ]
      },
      {
        $and: [
          { $ne: ['$serving_temp', null] },
          { $lt: ['$serving_temp', servingTempMin] }
        ]
      }
    ];
    if (useMissingForNg) {
      ngBranches.push({ $eq: ['$ingredient_missing', true] });
    }
    const statusExpr = {
      $cond: [
        { $or: ngBranches },
        'ng',
        'ok'
      ]
    };

    const pipe = [];
    if (Object.keys(match).length) pipe.push({ $match: match });

    pipe.push(
      {
        $project: {
          device_id: 1,
          dateStr: {
            $dateToString: {
              format: '%Y-%m-%d',
              date: '$event_time',
              timezone: tz
            }
          },
          production_time_sec: 1,
          serving_temp: 1,
          ingredient_missing: 1,
          status_norm: statusExpr
        }
      },
      // 關聯裝置 → 取得店鋪資訊
      {
        $lookup: {
          from: 'devices',
          localField: 'device_id',
          foreignField: 'device_id',
          as: 'dv'
        }
      },
      { $unwind: { path: '$dv', preserveNullAndEmptyArrays: includeUnknown } },
      {
        $project: {
          dateStr: 1,
          status_norm: 1,
          market:       includeUnknown ? { $ifNull: ['$dv.store.market', 'UNKNOWN'] } : '$dv.store.market',
          store_number: includeUnknown ? { $ifNull: ['$dv.store.number', 'UNKNOWN'] } : '$dv.store.number',
          store_name:   includeUnknown ? { $ifNull: ['$dv.store.name',   'UNKNOWN'] } : '$dv.store.name'
        }
      }
    );

    if (!includeUnknown) {
      pipe.push({
        $match: {
          market: { $ne: null },
          store_number: { $ne: null },
          store_name: { $ne: null }
        }
      });
    }

    pipe.push({
      $facet: {
        per_store: [
          {
            $group: {
              _id: { market: '$market', number: '$store_number', name: '$store_name' },
              ok_count: { $sum: { $cond: [{ $eq: ['$status_norm', okLc] }, 1, 0] } },
              total:    { $sum: 1 }
            }
          },
          {
            $project: {
              _id: 0,
              market:       '$_id.market',
              store_number: '$_id.number',
              store_name:   '$_id.name',
              ok_ratio: {
                $cond: [{ $gt: ['$total', 0] }, { $divide: ['$ok_count', '$total'] }, 0]
              }
            }
          },
          { $sort: { market: 1, store_number: 1, store_name: 1 } }
        ],
        daily_store_category: [
          {
            $group: {
              _id: { date: '$dateStr', market: '$market', number: '$store_number', name: '$store_name' },
              ok_count: { $sum: { $cond: [{ $eq: ['$status_norm', okLc] }, 1, 0] } },
              total:    { $sum: 1 }
            }
          },
          {
            $project: {
              date: '$_id.date',
              ratio: {
                $cond: [{ $gt: ['$total', 0] }, { $divide: ['$ok_count', '$total'] }, 0]
              }
            }
          },
          {
            $addFields: {
              category: {
                $switch: {
                  branches: [
                    { case: { $gt: ['$ratio', 0.9] }, then: 'excellent' },
                    {
                      case: {
                        $and: [
                          { $gte: ['$ratio', 0.8] },
                          { $lt:  ['$ratio', 0.9] }
                        ]
                      },
                      then: 'watch'
                    },
                    { case: { $lt: ['$ratio', 0.8] }, then: 'bad' }
                  ],
                  default: 'unknown'
                }
              }
            }
          },
          {
            $group: {
              _id: '$date',
              excellent_count: { $sum: { $cond: [{ $eq: ['$category', 'excellent'] }, 1, 0] } },
              watch_count:     { $sum: { $cond: [{ $eq: ['$category', 'watch']     }, 1, 0] } },
              bad_count:       { $sum: { $cond: [{ $eq: ['$category', 'bad']       }, 1, 0] } },
              total_stores:    { $sum: 1 }
            }
          },
          { $sort: { _id: 1 } },
          {
            $project: {
              _id: 0,
              date: '$_id',
              date_iso: { $concat: ['$_id', 'T00:00:00Z'] },
              excellent_count: 1,
              watch_count:     1,
              bad_count:       1,
              total_stores:    1
            }
          }
        ]
      }
    });

    const out = await col.aggregate(pipe).toArray();
    const payload = (out && out[0]) || { per_store: [], daily_store_category: [] };

    // 後面 summary 保持你原本的邏輯
    const validDevices = await mongoClient
      .db(MONGO_DB)
      .collection('devices')
      .distinct('device_id', { 'store.name': { $ne: null } });

    const prodMatch = {};
    if (tFrom || tTo) {
      prodMatch.event_time = {};
      if (tFrom) prodMatch.event_time.$gte = tFrom;
      if (tTo)   prodMatch.event_time.$lte = tTo;
    }
    prodMatch.device_id = { $in: validDevices };
    const total_productions = await col.countDocuments(prodMatch);

    const distinctStores = new Set(
      (payload.per_store || []).map(
        s => `${s.market || ''}|${s.store_number || ''}|${s.store_name || ''}`
      )
    );
    const total_stores = distinctStores.size;

    let excellent_count = 0, watch_count = 0, bad_count = 0;
    for (const s of payload.per_store || []) {
      const ratio = typeof s.ok_ratio === 'number' ? s.ok_ratio : 0;
      if (ratio > 0.9)               excellent_count += 1;
      else if (ratio >= 0.8 && ratio < 0.9) watch_count += 1;
      else                          bad_count += 1;
    }
    const period_store_category = {
      excellent_count,
      watch_count,
      bad_count,
      total_stores,
      excellent_ratio: total_stores ? excellent_count / total_stores : 0,
      watch_ratio:     total_stores ? watch_count     / total_stores : 0,
      bad_ratio:       total_stores ? bad_count       / total_stores : 0
    };

    return res.json({
      data: {
        per_store: payload.per_store || [],
        daily_store_category: payload.daily_store_category || []
      },
      summary: {
        all_stores_number: validDevices.length,
        total_stores,
        store_number_string: `${total_stores} / ${validDevices.length}`,
        total_productions,
        period_store_category
      }
    });
  } catch (err) {
    console.error('[GET /quality/store-ok-stats] error:', err);
    return res.status(500).json({ ok: false, error: String(err && err.message || err) });
  }
});


app.get('/quality/per-store-ok-ratio', authMiddleware, async (req, res) => { 
  try {
    const tFrom = parseISO(req.query.from);
    const tTo   = parseISO(req.query.to);
    const okLc  = String(req.query.ok_value || 'OK').toLowerCase();
    const tz    = String(req.query.tz || 'UTC');
    const includeUnknown = ['1','true','True'].includes(String(req.query.include_unknown || '0'));

    const productionLimitSec = Number.isFinite(Number(req.query.max_production_sec))
      ? Number(req.query.max_production_sec)
      : 120;
    const servingTempMin = Number.isFinite(Number(req.query.min_serving_temp))
      ? Number(req.query.min_serving_temp)
      : 75;

    // 食材缺失是否算 NG
    const missingAsNgRaw = String(req.query.missing_as_ng ?? '1').toLowerCase();
    const useMissingForNg = !['0', 'false', 'no'].includes(missingAsNgRaw);

    // ok_ratio 門檻
    const ratioLtRaw = req.query.ok_ratio_lt ?? req.query.threshold ?? req.query.ratio_lt;
    const ratioLt = Number.isFinite(Number(ratioLtRaw)) ? Number(ratioLtRaw) : null;
    const ratioGtRaw = req.query.ok_ratio_gt;
    const ratioGt = Number.isFinite(Number(ratioGtRaw)) ? Number(ratioGtRaw) : null;

    // 店號過濾
    let storeNumbers = null;
    if (req.query.store_number) {
      storeNumbers = String(req.query.store_number)
        .split(',')
        .map(s => s.trim())
        .filter(Boolean);
      if (!storeNumbers.length) storeNumbers = null;
    }
    const storeNumberLike = req.query.store_number_like
      ? new RegExp(String(req.query.store_number_like).trim(), 'i')
      : null;

    const match = {};
    if (tFrom || tTo) {
      match.event_time = {};
      if (tFrom) match.event_time.$gte = tFrom;
      if (tTo)   match.event_time.$lte = tTo;
    }

    // 組 status_norm expression
    const ngBranches = [
      {
        $and: [
          { $ne: ['$production_time_sec', null] },
          { $gt: ['$production_time_sec', productionLimitSec] }
        ]
      },
      {
        $and: [
          { $ne: ['$serving_temp', null] },
          { $lt: ['$serving_temp', servingTempMin] }
        ]
      }
    ];
    if (useMissingForNg) {
      ngBranches.push({ $eq: ['$ingredient_missing', true] });
    }
    const statusExpr = {
      $cond: [
        { $or: ngBranches },
        'ng',
        'ok'
      ]
    };

    const pipe = [];
    if (Object.keys(match).length) pipe.push({ $match: match });

    pipe.push(
      {
        $project: {
          device_id: 1,
          dateStr: {
            $dateToString: {
              format: '%Y-%m-%d',
              date: '$event_time',
              timezone: tz
            }
          },
          production_time_sec: 1,
          serving_temp: 1,
          ingredient_missing: 1,
          status_norm: statusExpr
        }
      },
      {
        $lookup: {
          from: 'devices',
          localField: 'device_id',
          foreignField: 'device_id',
          as: 'dv'
        }
      },
      { $unwind: { path: '$dv', preserveNullAndEmptyArrays: includeUnknown } },
      {
        $project: {
          dateStr: 1,
          status_norm: 1,
          market:       includeUnknown ? { $ifNull: ['$dv.store.market', 'UNKNOWN'] } : '$dv.store.market',
          store_number: includeUnknown ? { $ifNull: ['$dv.store.number', 'UNKNOWN'] } : '$dv.store.number',
          store_name:   includeUnknown ? { $ifNull: ['$dv.store.name',   'UNKNOWN'] } : '$dv.store.name',
          latitude:     '$dv.store.latitude',
          longitude:    '$dv.store.longitude'
        }
      }
    );

    if (!includeUnknown) {
      pipe.push({
        $match: {
          market: { $ne: null },
          store_number: { $ne: null },
          store_name: { $ne: null }
        }
      });
    }

    // 店號過濾
    if (storeNumbers && storeNumberLike) {
      pipe.push({
        $match: {
          $or: [
            { store_number: { $in: storeNumbers } },
            { store_number: storeNumberLike }
          ]
        }
      });
    } else if (storeNumbers) {
      pipe.push({ $match: { store_number: { $in: storeNumbers } } });
    } else if (storeNumberLike) {
      pipe.push({ $match: { store_number: storeNumberLike } });
    }

    const perStoreStages = [
      {
        $group: {
          _id: {
            market: '$market',
            number: '$store_number',
            name: '$store_name'
          },
          ok_count: { $sum: { $cond: [{ $eq: ['$status_norm', okLc] }, 1, 0] } },
          total:    { $sum: 1 },
          latitude:  { $first: '$latitude' },
          longitude: { $first: '$longitude' }
        }
      },
      {
        $project: {
          _id: 0,
          market:       '$_id.market',
          store_number: '$_id.number',
          store_name:   '$_id.name',
          latitude:     1,
          longitude:    1,
          ok_ratio: {
            $cond: [
              { $gt: ['$total', 0] },
              { $divide: ['$ok_count', '$total'] },
              0
            ]
          }
        }
      }
    ];

    if (ratioLt !== null) {
      perStoreStages.push({ $match: { ok_ratio: { $lt: ratioLt } } });
    }
    if (ratioGt !== null) {
      perStoreStages.push({ $match: { ok_ratio: { $gte: ratioGt } } });
    }

    perStoreStages.push({ $sort: { ok_ratio: 1 } });

    pipe.push({ $facet: { per_store: perStoreStages } });

    const out = await col.aggregate(pipe).toArray();
    const payload = (out && out[0]) || { per_store: [] };

    return res.json({
      data: {
        per_store: payload.per_store || []
      }
    });
  } catch (err) {
    console.error('[GET /quality/per-store-ok-ratio] error:', err);
    return res.status(500).json({ ok: false, error: String(err && err.message || err) });
  }
});

// GET /quality/store-stats?store_number=SH219&from=2025-10-01T00:00:00Z&to=2025-10-31T23:59:59Z
// 回傳：該店在期間內的合格率、溫度不足率、製作超時率、食材缺失率、總銷售數 + 店鋪資訊
app.get('/quality/store-stats', async (req, res) => {
  try {
    const storeNumber = String(req.query.store_number || '').trim();
    if (!storeNumber) {
      return res.status(400).json({ ok: false, error: 'store_number is required' });
    }
    const tFrom = parseISO(req.query.from);
    const tTo   = parseISO(req.query.to);

    // 動態門檻
    const productionLimitSec = Number.isFinite(Number(req.query.max_production_sec))
      ? Number(req.query.max_production_sec)
      : 120;  // 預設：>120 秒算超時 / NG

    const servingTempMin = Number.isFinite(Number(req.query.min_serving_temp))
      ? Number(req.query.min_serving_temp)
      : 75;   // 預設：<75 度算低溫 / NG

    // 食材缺失是否要算 NG 條件 ＋ 是否要統計
    const missingAsNgRaw = String(req.query.missing_as_ng ?? '1').toLowerCase();
    const useMissingForNg = !['0', 'false', 'no'].includes(missingAsNgRaw);
    const disableMissingStat = !useMissingForNg; // 關掉時 missing_count / ratio 都要視為 0

    // 先找出這個店號對應的所有 device_id
    const storeQuery = { 'store.number': storeNumber };
    const deviceIds = await devCol.distinct('device_id', storeQuery);

    // 取店鋪資訊（用第一筆）
    const storeDoc = await devCol.findOne(storeQuery, {
      projection: { _id: 0, store: 1, device_id: 1 }
    });

    if (!deviceIds.length) {
      return res.json({
        ok: true,
        data: {
          store: storeDoc ? {
            number:   storeDoc.store?.number || storeNumber,
            name:     storeDoc.store?.name   || null,
            market:   storeDoc.store?.market || null,
            latitude: storeDoc.store?.latitude ?? null,
            longitude:storeDoc.store?.longitude ?? null
          } : { number: storeNumber, name: null, market: null, latitude: null, longitude: null },
          window: { from: tFrom || null, to: tTo || null },
          totals: { total: 0, ok_count: 0, low_temp_count: 0, overtime_count: 0, missing_count: 0 },
          ratios: { ok_ratio: 0, low_temp_ratio: 0, overtime_ratio: 0, ingredient_missing_ratio: 0 }
        }
      });
    }

    const match = { device_id: { $in: deviceIds } };
    if (tFrom || tTo) {
      match.event_time = {};
      if (tFrom) match.event_time.$gte = tFrom;
      if (tTo)   match.event_time.$lte = tTo;
    }

    const rows = await col.aggregate([
      { $match: match },

      // 先把各種條件算成 boolean，再算 status_norm
      {
        $project: {
          production_time_sec: 1,
          serving_temp: 1,
          ingredient_missing: 1,

          is_overtime_for_ng: {
            $and: [
              { $ne: ['$production_time_sec', null] },
              { $gt: ['$production_time_sec', productionLimitSec] }
            ]
          },
          is_low_temp: {
            $and: [
              { $ne: ['$serving_temp', null] },
              { $lt: ['$serving_temp', servingTempMin] }
            ]
          },
          is_missing: { $eq: ['$ingredient_missing', true] },

          status_norm: {
            $cond: [
              {
                $or: [
                  // 超時
                  {
                    $and: [
                      { $ne: ['$production_time_sec', null] },
                      { $gt: ['$production_time_sec', productionLimitSec] }
                    ]
                  },
                  // 低溫
                  {
                    $and: [
                      { $ne: ['$serving_temp', null] },
                      { $lt: ['$serving_temp', servingTempMin] }
                    ]
                  },
                  // 食材缺失（可開關）
                  {
                    $and: [
                      { $literal: useMissingForNg },
                      { $eq: ['$ingredient_missing', true] }
                    ]
                  }
                ]
              },
              'ng',
              'ok'
            ]
          }
        }
      },

      // 統計
      {
        $group: {
          _id: null,
          total: { $sum: 1 },
          ok_count: {
            $sum: { $cond: [{ $eq: ['$status_norm', 'ok'] }, 1, 0] }
          },
          low_temp_count: {
            $sum: {
              $cond: ['$is_low_temp', 1, 0]
            }
          },
          overtime_count: {
            $sum: {
              $cond: ['$is_overtime_for_ng', 1, 0]
            }
          },
          // 先把「真實缺料數」算起來，等一下再決定要不要歸零
          missing_count_raw: {
            $sum: {
              $cond: ['$is_missing', 1, 0]
            }
          }
        }
      },

      // 比例＋根據 missing_as_ng 決定要不要清零
      {
        $project: {
          _id: 0,
          total: 1,
          ok_count: 1,
          low_temp_count: 1,
          overtime_count: 1,

          // missing_count：missing_as_ng=0 → 0
          missing_count: {
            $cond: [
              { $literal: disableMissingStat },
              0,
              '$missing_count_raw'
            ]
          },

          ok_ratio: {
            $cond: [
              { $gt: ['$total', 0] },
              { $divide: ['$ok_count', '$total'] },
              0
            ]
          },
          low_temp_ratio: {
            $cond: [
              { $gt: ['$total', 0] },
              { $divide: ['$low_temp_count', '$total'] },
              0
            ]
          },
          overtime_ratio: {
            $cond: [
              { $gt: ['$total', 0] },
              { $divide: ['$overtime_count', '$total'] },
              0
            ]
          },

          // ingredient_missing_ratio：missing_as_ng=0 → 0
          ingredient_missing_ratio: {
            $cond: [
              { $literal: disableMissingStat },
              0,
              {
                $cond: [
                  { $gt: ['$total', 0] },
                  {
                    $divide: [
                      '$missing_count_raw',
                      '$total'
                    ]
                  },
                  0
                ]
              }
            ]
          }
        }
      }
    ]).toArray();

    const agg = rows[0] || {
      total: 0, ok_count: 0, low_temp_count: 0, overtime_count: 0, missing_count: 0,
      ok_ratio: 0, low_temp_ratio: 0, overtime_ratio: 0, ingredient_missing_ratio: 0
    };

    const fromStr = tFrom ? tFrom.toISOString().split('T')[0] : null;
    const toStr   = tTo   ? tTo.toISOString().split('T')[0]   : null;

    return res.json({
      ok: true,
      data: {
        store: {
          number: storeNumber,
          name: storeDoc?.store?.name || null,
          market: storeDoc?.store?.market || null,
          latitude: storeDoc?.store?.latitude ?? null,
          longitude: storeDoc?.store?.longitude ?? null
        },
        window: { from: fromStr, to: toStr },
        totals: {
          total: agg.total,
          ok_count: agg.ok_count,
          low_temp_count: agg.low_temp_count,
          overtime_count: agg.overtime_count,
          missing_count: agg.missing_count
        },
        ratios: {
          ok_ratio: agg.ok_ratio,
          low_temp_ratio: agg.low_temp_ratio,
          overtime_ratio: agg.overtime_ratio,
          ingredient_missing_ratio: agg.ingredient_missing_ratio,
          score_ratio: 0
        }
      }
    });
  } catch (err) {
    console.error('[GET /quality/store-stats] error:', err);
    return res.status(500).json({ ok: false, error: String(err && err.message || err) });
  }
});


// ===== helper：彈性時間解析（ISO / unix秒 / unix毫秒） =====
function parseFlexibleTime(t) {
  if (!t) return null;
  const s = String(t).trim();
  if (/^\d+$/.test(s)) {
    const num = Number(s);
    if (num > 1e12) return new Date(num);        // 毫秒 timestamp (e.g. 1762198486000)
    if (num > 1e9)  return new Date(num * 1000); // 秒 timestamp   (e.g. 1762198486)
  }
  try { return parseISO(s); } catch { return null; }
}

// ===== helper：把 raw 裡的溫度欄位轉成數字（空字串/null → null）=====
const toNum = (path) => ({
  $cond: [
    { $in: [ { $type: `$${path}` }, ['double','int','long','decimal'] ] },
    { $toDouble: `$${path}` },
    {
      $cond: [
        { $or: [ { $eq: [ `$${path}`, '' ] }, { $eq: [ `$${path}`, null ] } ] },
        null,
        { $toDouble: `$${path}` }
      ]
    }
  ]
});

// 你目前 ingest 會用到的加料欄位（值為 "1" 代表有加）
const TOPPING_KEYS = [
  'ajisen_pork','chill_beef','miyazaki_meat','pork_jowl','pork_neck',
  'tomato_beef','tonkotsu','bean_sprout','black_fungus','cabbage',
  'corn','egg','garlic','menma','noodle','soup','tomato','green_onions','coriander'
];
const TOPPING_LABEL_MAP = {
  ajisen_pork: "味千叉烧",
  chill_beef: "冷冻牛肉",
  miyazaki_meat: "宫崎牛",
  pork_jowl: "猪颊肉",
  pork_neck: "猪颈肉",
  tomato_beef: "番茄牛肉",
  tonkotsu: "猪软骨",
  bean_sprout: "豆芽菜",
  black_fungus: "木耳",
  cabbage: "高丽菜",
  corn: "玉米",
  egg: "溏心蛋",
  garlic: "蒜泥",
  menma: "笋乾",
  noodle: "面",
  soup: "汤",
  tomato: "番茄",
  green_onions: "葱花",
  coriander: "香菜"
};
// 對照表
const RAMEN_NAME_MAP = {
  0: '经典味千拉面',
  1: '宫崎辛面',
  2: '大骨浓汤猪软骨拉面',
  3: '麻辣牛肉拉面',
  4: '番茄肥牛拉面',
  5: '浓厚骨汤二郎拉面',
  6: '无法识别',
  8: '新菜单',
  9: '虎虾拉面',
  10: '鬼金棒拉面',
  11: '菌菇拉面'
};

function normalizeRamenName(rawVal) {
  if (rawVal === null || rawVal === undefined) return null;

  const s = String(rawVal).trim();
  if (!s) return null;

  // 若是純數字，就用對照表
  if (/^\d+$/.test(s)) {
    const idx = parseInt(s, 10);
    return RAMEN_NAME_MAP[idx] ?? RAMEN_NAME_MAP[6]; // 超出 0~5 一律當「無法識別」
  }

  // 否則直接用原字串
  return s;
}

// ===== API：期間內某店鋪的製作明細（含 toppings 與 ramen_name） =====
// GET /quality/store-productions?store_number=SH219
//                                [&from=2025-10-01T00:00:00Z]
//                                [&to=2025-10-31T23:59:59Z]
//                                [&status=ok|ng]
//                                [&event_time=1762198486000|2025-10-14T10:18:26Z]
//                                [&window_ms=1000]   // 單點視窗，預設 1000ms
app.get('/quality/store-productions', async (req, res) => {
  try {
    const storeNumber = String(req.query.store_number || '').trim();
    if (!storeNumber) {
      return res.status(400).json({ ok: false, error: 'store_number is required' });
    }

    // 可能的單筆查詢方式
    const idRaw = req.query.id || req.query._id;
    let idObj = null;
    if (idRaw) {
      try { idObj = new ObjectId(String(idRaw)); }
      catch (e) { return res.status(400).json({ ok: false, error: 'invalid _id format' }); }
    }

    // 時間條件
    const tFrom = parseFlexibleTime(req.query.from);
    const tTo   = parseFlexibleTime(req.query.to);
    const tAt   = parseFlexibleTime(req.query.event_time); // 單點
    const windowMs = Number.isFinite(Number(req.query.window_ms))
      ? Number(req.query.window_ms) : 1000;

    // 合格與否
    const status = req.query.status ? String(req.query.status).toLowerCase() : null; // 'ok' | 'ng'

    // 找出該店號所有裝置
    const deviceIds = await devCol.distinct('device_id', { 'store.number': storeNumber });
    if (!deviceIds.length) {
      return res.json({ ok: true, data: [], note: `no devices for store_number=${storeNumber}` });
    }

    // 組合 match 條件
    const match = { device_id: { $in: deviceIds } };

    if (idObj) {
      // ✅ 若有傳 _id，就優先直接查指定那筆，不再用 event_time 篩選
      match._id = idObj;
    } else if (tAt) {
      const tAtEnd = new Date(tAt.getTime() + windowMs);
      match.event_time = { $gte: tAt, $lt: tAtEnd };
    } else if (tFrom || tTo) {
      match.event_time = {};
      if (tFrom) match.event_time.$gte = tFrom;
      if (tTo)   match.event_time.$lte = tTo;
    }

    if (status === 'ok' || status === 'ng') {
      match.status_norm = status;
    }

    // toppings 陣列：取 raw.<key> == "1" 的項目
    // const toppingsArray = {
    //   $map: {
    //     input: TOPPING_KEYS,
    //     as: "t",
    //     in: {
    //       $cond: [
    //         { $eq: [ { $toString: `$raw.$$[t]` }, "1" ] },
    //         TOPPING_LABEL_MAP["$$t"],   // ← 對照表轉中文
    //         null
    //       ]
    //     }
    //   }
    // };

    const rows = await col.aggregate([
      { $match: match },
      {
        $project: {
          _id: 0,
          time: { $dateToString: { format: '%Y-%m-%dT%H:%M:%SZ', date: '$event_time', timezone: 'UTC' } },
      
          production_time_sec: '$production_time_sec',
          serving_temp: '$serving_temp',
          status_norm: 1,
      
          empty_bowl_temp: toNum('raw.Empty Bowl Temp'),
          soup_temp: toNum('raw.Soup Temp'),
          noodle_temp: toNum('raw.Noodle Temp'),
      
          ramen_name: { $ifNull: ['$raw.ramen_name', null] },
      
          toppings: {
            $filter: {
              input: TOPPING_KEYS.map(k => ({
                $cond: [
                  { $eq: [ { $toString: `$raw.${k}` }, "1" ] },
                  TOPPING_LABEL_MAP[k],  // ✅ 英文 key → 中文名稱
                  null
                ]
              })),
              as: 'x',
              cond: { $ne: ['$$x', null] }
            }
          }
        }
      },
      { $sort: { time: 1 } }
    ]).toArray();

    // 這裡把數字代碼轉可讀的名稱
    const mappedRows = rows.map(r => ({
      ...r,
      ramen_name: normalizeRamenName(r.ramen_name)
    }));

    return res.json({ ok: true, data: mappedRows });
  } catch (err) {
    console.error('[GET /quality/store-productions] error:', err);
    return res.status(500).json({ ok: false, error: String(err && err.message || err) });
  }
});


// === 單點圖片查詢：用 store_number + event_time 找五張圖 ===
// GET /quality/store-production-images?store_number=BJ118
//     &event_time=1762198486000
//     [&window_ms=1000]
app.get('/quality/store-production-images', async (req, res) => {
  try {
    const imageType = req.query.imageType;
    // 可能的單筆查詢方式
    const idRaw = req.query.id || req.query._id;
    let idObj = null;
    if (idRaw) {
      try { idObj = new ObjectId(String(idRaw)); }
      catch (e) { return res.status(400).json({ ok: false, error: 'invalid _id format' }); }
    }

    const storeNumber = String(req.query.store_number || '').trim();
    if (!storeNumber) return res.status(400).json({ ok: false, error: 'store_number is required' });

    // 單點時間（支援 ISO / unix 秒 / unix 毫秒）
    const tAt = parseFlexibleTime(req.query.event_time);
    // if (!tAt) return res.status(400).json({ ok: false, error: 'event_time is required (ISO/seconds/ms)' });
    // const windowMs = Number.isFinite(Number(req.query.window_ms)) ? Number(req.query.window_ms) : 1000;
    // const tAtEnd = new Date(tAt.getTime() + windowMs);

    // 找店號所有 device_id
    const deviceIds = await devCol.distinct('device_id', { 'store.number': storeNumber });
    if (!deviceIds.length) return res.json({ ok: true, data: [], note: `no devices for store_number=${storeNumber}` });

    let match = {
      device_id: { $in: deviceIds }
    };
    if (idObj) {
      // ✅ 若有傳 _id，就優先直接查指定那筆，不再用 event_time 篩選
      match._id = idObj;
    }else if (tAt) {
      const tAtEnd = new Date(tAt.getTime() + windowMs);
      match.event_time = { $gte: tAt, $lt: tAtEnd };
    }

    let project = {
      _id: 0,
      time: { $dateToString: { format: '%Y-%m-%dT%H:%M:%SZ', date: '$event_time', timezone: 'UTC' } },
    };
    if(imageType == 1) {
      project.empty_bowl_image = '$raw.empty_bowl_image';
    }else if(imageType == 2){
      project.soup_image = '$raw.soup_image';
    }else if(imageType == 3){
      project.noodle_image = '$raw.noodle_image';
    }else if(imageType == 5){
      project.serving_front_image = '$raw.serving_front_image';
    }else{
      project.serving_back_image = '$raw.serving_back_image';
    }

    const rows = await col.aggregate([
      { $match: match },
      { $project: project},
      { $sort: { time: 1 } }
    ]).toArray();

    // 若同一秒有多筆，回傳陣列；通常只會有一筆
    return res.json({ ok: true, data: rows });
  } catch (err) {
    console.error('[GET /quality/store-production-images] error:', err);
    return res.status(500).json({ ok: false, error: String(err && err.message || err) });
  }
});
 
app.get('/quality/thresholds', async (req, res) => {
  try {
    console.log('/quality/thresholds')
    console.log(req.user)
    const doc = await settingCol.findOne({ _id: 'quality_thresholds' });

    const out = {
      max_production_sec: doc?.max_production_sec ?? DEFAULT_MAX_PRODUCTION_SEC,
      min_serving_temp:   doc?.min_serving_temp   ?? DEFAULT_MIN_SERVING_TEMP,
      missing_as_ng:      typeof doc?.missing_as_ng === 'boolean'
        ? doc.missing_as_ng
        : DEFAULT_MISSING_AS_NG
    };

    return res.json({ ok: true, data: out });
  } catch (err) {
    console.error('[GET /quality/thresholds] error:', err);
    return res.status(500).json({ ok: false, error: String(err && err.message || err) });
  }
});

app.post('/quality/thresholds', async (req, res) => {
  try {
    const body = req.body || {};

    let maxProduction = Number(body.max_production_sec);
    let minServing    = Number(body.min_serving_temp);
    let missingAsNg   = parseBoolLoose(body.missing_as_ng, DEFAULT_MISSING_AS_NG);

    if (!Number.isFinite(maxProduction) || maxProduction <= 0) {
      maxProduction = DEFAULT_MAX_PRODUCTION_SEC;
    }
    if (!Number.isFinite(minServing) || minServing <= 0) {
      minServing = DEFAULT_MIN_SERVING_TEMP;
    }

    const updateDoc = {
      max_production_sec: maxProduction,
      min_serving_temp:   minServing,
      missing_as_ng:      missingAsNg
    };

    await settingCol.updateOne(
      { _id: 'quality_thresholds' },
      { $set: updateDoc },
      { upsert: true }
    );

    return res.json({ ok: true, data: updateDoc });
  } catch (err) {
    console.error('[POST /quality/thresholds] error:', err);
    return res.status(500).json({ ok: false, error: String(err && err.message || err) });
  }
});


// 簡單帳號密碼（之後可改成讀環境變數或 DB）
const USERS = {
  admin: 'ajisen123',   // 帳號: admin, 密碼: ajisen123
  viewer: 'ramen456',
  Primax01: 'xS7#dBaU7abF*!3ePOsf$mv37qkLYpnL'
};
// { token: { username, createdAt } }
const TOKENS = new Map();

const TOKEN_TTL_MS = 8 * 60 * 60 * 1000; // 2 小時

// POST /quality/login
app.post('/quality/login', (req, res) => {
  const { username, password } = req.body || {};

  if (!username || !password) {
    return res.status(400).json({ ok: false, error: 'missing username or password' });
  }

  const expected = USERS[username];
  if (!expected || expected !== password) {
    return res.status(401).json({ ok: false, error: 'invalid credentials' });
  }

  // ✅ 產生 token
  const token = crypto.randomBytes(32).toString('hex');

  TOKENS.set(token, {
    username,
    createdAt: Date.now()
  });

  return res.json({
    ok: true,
    user: { username },
    token
  });
});

function authMiddleware(req, res, next) {
  const token = req.headers["x-ajisen-token"];
  // const auth = req.headers['authorization'] || '';
  // const token = auth.startsWith('Bearer ') ? auth.slice(7) : null;

  if (!token) {
    return res.status(401).json({ ok: false, error: 'missing token' });
  }

  const record = TOKENS.get(token);
  if (!record) {
    return res.status(401).json({ ok: false, error: 'invalid token' });
  }

  // 檢查過期
  if (Date.now() - record.createdAt > TOKEN_TTL_MS) {
    TOKENS.delete(token);
    return res.status(401).json({ ok: false, error: 'token expired' });
  }

  // 綁定使用者
  req.user = record.username;
  req.token = token;

  next();
}

// GET /quality/auth-check
app.get('/quality/auth-check', authMiddleware, (req, res) => {
  return res.json({
    ok: true,
    user: {
      username: req.user
    }
  });
});


//=============================================================================================================== 
/////////////////////////////////////////////////////////////////new////////////////////////////////////////////////
//=============================================================================================================== 



// settings collection：建議 quality_settings
// 你可以用同一個 DB，另開 collection: quality_settings
// const col_settings = db.collection('quality_settings');
function defaultSettings() {
  return {
    type: "quality_settings",
    ramen_standard: {
      min_temp_c: 85,
      max_production_time_sec: 120,
    },
    store_grading: {
      excellent_ratio_min: 0.90,
      watch_ratio_min: 0.80,
      bad_ratio_max: 0.80,
    },
    meal_periods: [
      { key: "morning_peak", name: "早低峰", enabled: true, start: "09:00", end: "12:00" },
      { key: "noon_peak", name: "午高峰", enabled: true, start: "12:00", end: "14:00" },
      { key: "afternoon_peak", name: "午後峰", enabled: true, start: "14:00", end: "17:00" },
      { key: "evening_peak", name: "晚高峰", enabled: true, start: "17:00", end: "19:00" },
      { key: "late_peak", name: "晚低峰", enabled: true, start: "19:00", end: "21:00" },
      { key: "special_1", name: "特殊時段1", enabled: true, start: "04:00", end: "06:00" },
      { key: "special_2", name: "特殊時段2", enabled: true, start: "06:00", end: "09:00" },
    ],
    data_retention: { period: "season" },
    updated_at: new Date(),
  };
}

// ---- 工具：簡易驗證（避免亂填）----

function hhmmToMin(hhmm) {
  const [h, m] = hhmm.split(':').map(n => parseInt(n, 10));
  return h * 60 + m;
}

function isHHMM(s) {
  return typeof s === "string" && /^[0-2]\d:[0-5]\d$/.test(s);
}

function clamp01(x) {
  const n = Number(x);
  if (!Number.isFinite(n)) return null;
  return Math.max(0, Math.min(1, n));
}

function parseMulti(val, { allowAll = false } = {}) {
  if (val == null) return null;
  let arr = null;
  if (Array.isArray(val)) arr = val.map(x => String(x).trim()).filter(Boolean);
  else arr = String(val).split(',').map(s => s.trim()).filter(Boolean);
  if (!arr.length) return null;
  if (allowAll && arr.some(x => x.toLowerCase() === 'all')) return null;
  return arr;
}

app.post('/login', (req, res) => {
  const { username, password } = req.body || {};

  if (!username || !password) {
    return res.status(400).json({ ok: false, error: 'missing username or password' });
  }

  const expected = USERS[username];
  if (!expected || expected !== password) {
    return res.status(401).json({ ok: false, error: 'invalid credentials' });
  }

  // ✅ 產生 token
  const token = crypto.randomBytes(32).toString('hex');

  TOKENS.set(token, {
    username,
    createdAt: Date.now()
  });

  return res.json({
    ok: true,
    user: { username },
    token
  });
});

// ========== 取得 ==========
app.post("/settings/get", async (req, res) => {
  try {
    const doc = await settingCol.findOne({ type: "quality_settings" });
    if (!doc) return res.json({ ok: true, data: defaultSettings() });
    return res.json({ ok: true, data: doc });
  } catch (e) {
    return res.status(500).json({ ok: false, error: e.message || String(e) });
  }
});

// ========== 設定（更新/覆蓋） ==========
app.post("/settings/set", async (req, res) => {
  try {
    const body = req.body || {};
    const patch = {};

    // ramen_standard
    if (body.ramen_standard) {
      const rs = body.ramen_standard;
      if (rs.min_temp_c != null) {
        const t = Number(rs.min_temp_c);
        if (!Number.isFinite(t) || t < 0 || t > 120) {
          return res.status(400).json({ ok: false, error: "min_temp_c invalid" });
        }
        patch["ramen_standard.min_temp_c"] = t;
      }
      if (rs.max_production_time_sec != null) {
        const s = Number(rs.max_production_time_sec);
        if (!Number.isFinite(s) || s < 1 || s > 36000) {
          return res.status(400).json({ ok: false, error: "max_production_time_sec invalid" });
        }
        patch["ramen_standard.max_production_time_sec"] = s;
      }
    }

    // store_grading
    if (body.store_grading) {
      const g = body.store_grading;
      if (g.excellent_ratio_min != null) {
        const v = clamp01(g.excellent_ratio_min);
        if (v == null) return res.status(400).json({ ok: false, error: "excellent_ratio_min invalid" });
        patch["store_grading.excellent_ratio_min"] = v;
      }
      if (g.watch_ratio_min != null) {
        const v = clamp01(g.watch_ratio_min);
        if (v == null) return res.status(400).json({ ok: false, error: "watch_ratio_min invalid" });
        patch["store_grading.watch_ratio_min"] = v;
      }
      if (g.bad_ratio_max != null) {
        const v = clamp01(g.bad_ratio_max);
        if (v == null) return res.status(400).json({ ok: false, error: "bad_ratio_max invalid" });
        patch["store_grading.bad_ratio_max"] = v;
      }
    }

    // meal_periods（整包覆蓋最簡單、最不容易出現殘值）
    if (Array.isArray(body.meal_periods)) {
      for (const p of body.meal_periods) {
        if (!p.key || typeof p.key !== "string") {
          return res.status(400).json({ ok: false, error: "meal_periods.key required" });
        }
        if (p.start && !isHHMM(p.start)) return res.status(400).json({ ok: false, error: `meal_periods[${p.key}].start invalid` });
        if (p.end && !isHHMM(p.end)) return res.status(400).json({ ok: false, error: `meal_periods[${p.key}].end invalid` });
      }
      patch["meal_periods"] = body.meal_periods;
    }

    // data_retention
    if (body.data_retention?.period != null) {
      const allowed = new Set(["week", "month", "quarter", "season", "year"]);
      const v = String(body.data_retention.period);
      if (!allowed.has(v)) return res.status(400).json({ ok: false, error: "data_retention.period invalid" });
      patch["data_retention.period"] = v;
    }

    patch["updated_at"] = new Date();

    if (Object.keys(patch).length === 1 && patch.updated_at) {
      return res.status(400).json({ ok: false, error: "no valid fields to update" });
    }

    await settingCol.updateOne(
      { type: "quality_settings" },
      { $set: patch, $setOnInsert: { type: "quality_settings" } },
      { upsert: true }
    );

    const doc = await settingCol.findOne({ type: "quality_settings" });
    return res.json({ ok: true, data: doc });
  } catch (e) {
    return res.status(500).json({ ok: false, error: e.message || String(e) });
  }
});


// =====================================================
// POST /devices/markets
// 取得所有區域 (market_id, market) + store_count
// =====================================================
app.post('/listMarkets', async (req, res) => {
  try {
    const pipe = [
      {
        $match: {
          'store.market_id': { $ne: null },
          'store.market': { $ne: null },
        }
      },
      {
        $group: {
          _id: { market_id: '$store.market_id', market: '$store.market' },
          store_count: { $addToSet: '$store.number' }
        }
      },
      {
        $project: {
          _id: 0,
          market_id: '$_id.market_id',
          market: '$_id.market',
          // store_count: { $size: '$store_count' }
        }
      },
      { $sort: { market_id: 1 } }
    ];

    const data = await devCol.aggregate(pipe).toArray();
    return res.json({ ok: true, data });
  } catch (err) {
    console.error('[POST /devices/markets] error:', err);
    return res.status(500).json({ ok: false, error: String(err?.message || err) });
  }
});

// =====================================================
// POST /devices/stores
// 取得店鋪(name, number)，可用 market_id 過濾
// =====================================================
app.post('/listStores', async (req, res) => {
  try {
    const body = req.body || {};

    // market_id 支援：字串 / 陣列 / csv
    let marketIds = null;
    if (body.market_id != null) {
      if (Array.isArray(body.market_id)) {
        marketIds = body.market_id.map(x => String(x).trim()).filter(Boolean);
      } else {
        marketIds = String(body.market_id).split(',').map(s => s.trim()).filter(Boolean);
      }
      if (!marketIds.length) marketIds = null;
    }

    const match = {
      'store.number': { $ne: null },
      'store.name': { $ne: null }
    };
    if (marketIds) match['store.market_id'] = { $in: marketIds };

    const pipe = [
      { $match: match },
      // 有些資料可能重複寫入，用 number 去重
      {
        $group: {
          _id: '$store.number',
          name: { $first: '$store.name' },
          number: { $first: '$store.number' },
          market: { $first: '$store.market' },
          market_id: { $first: '$store.market_id' },
          // latitude: { $first: '$store.latitude' },
          // longitude: { $first: '$store.longitude' },
        }
      },
      {
        $project: {
          _id: 0,
          name: 1,
          number: 1,
          market: 1,
          market_id: 1,
          // latitude: 1,
          // longitude: 1
        }
      },
      { $sort: { market_id: 1, number: 1 } }
    ];

    const data = await devCol.aggregate(pipe).toArray();
    return res.json({ ok: true, data });
  } catch (err) {
    console.error('[POST /devices/stores] error:', err);
    return res.status(500).json({ ok: false, error: String(err?.message || err) });
  }
});

// =====================================================
// POST /quality/store-ok-stats  (完整整合版)
// =====================================================
// =====================================================
// 主 API
// 依你的專案：
//   col        = productions collection（有 event_time, device_id, serving_temp, production_time_sec, ingredient_missing）
//   deviceCol  = devices collection（有 store.market_id / store.number / store.name / lat / lon）
//   settingCol = settings collection（有 type=quality_settings）
//   authMiddleware 已存在
// =====================================================

app.post('/store-ok-stats', async (req, res) => {
  try {
    const body = req.body || {};

    // -----------------------------
    // 基本參數
    // -----------------------------
    const tFrom = parseISO(body.from);
    const tTo   = parseISO(body.to);
    const tz    = String(body.tz || 'Asia/Shanghai');
    const includeUnknown = ['1', 'true', 'True'].includes(String(body.include_unknown || '0'));

    // market_id filter: string | array | csv
    let marketIds = null;
    if (body.market_id != null) {
      if (Array.isArray(body.market_id)) {
        marketIds = body.market_id.map(x => String(x).trim()).filter(Boolean);
      } else {
        marketIds = String(body.market_id).split(',').map(s => s.trim()).filter(Boolean);
      }
      if (!marketIds.length) marketIds = null;
    }

    // -----------------------------
    // 讀 settings：quality_settings（依你提供的 qsDoc）
    // -----------------------------
    const qsDoc = await settingCol.findOne({ type: 'quality_settings' });
    if (!qsDoc) {
      return res.status(500).json({ ok: false, error: 'quality_settings not found in settings collection' });
    }

    // ramen_standard 門檻
    const minServingTemp = Number(qsDoc?.ramen_standard?.min_temp_c);
    const maxProductionSec = Number(qsDoc?.ramen_standard?.max_production_time_sec);
    if (!Number.isFinite(minServingTemp) || !Number.isFinite(maxProductionSec)) {
      return res.status(500).json({
        ok: false,
        error: 'quality_settings.ramen_standard invalid (min_temp_c / max_production_time_sec)'
      });
    }

    // 是否把缺料當 NG：你目前 qsDoc 沒這欄，先固定 true
    const missingAsNg = true;

    // store_grading 門檻
    const excellentMin = Number(qsDoc?.store_grading?.excellent_ratio_min);
    const watchMin = Number(qsDoc?.store_grading?.watch_ratio_min);
    const badMax = Number(qsDoc?.store_grading?.bad_ratio_max);

    if (!Number.isFinite(excellentMin) || !Number.isFinite(watchMin)) {
      return res.status(500).json({
        ok: false,
        error: 'quality_settings.store_grading invalid (excellent_ratio_min / watch_ratio_min)'
      });
    }

    // 一致性提醒（不影響執行）
    if (Number.isFinite(badMax) && Math.abs(badMax - watchMin) > 1e-9) {
      console.warn('[quality_settings] bad_ratio_max != watch_ratio_min, using watch_ratio_min as boundary');
    }

    function gradeByOkRatio(okRatio) {
      const r = Number.isFinite(Number(okRatio)) ? Number(okRatio) : 0;
      if (r >= excellentMin) return 'excellent';
      if (r >= watchMin) return 'watch';
      return 'bad';
    }

    // meal periods（完全用 qsDoc.meal_periods）
    const mealPeriods = Array.isArray(qsDoc?.meal_periods) ? qsDoc.meal_periods : [];

    // -----------------------------
    // meal_period_key filter (可複選)
    // 支援：
    //   "noon_peak"
    //   ["noon_peak","evening_peak"]
    //   "noon_peak,evening_peak"
    //   "all" / null -> 不過濾
    // -----------------------------
    let mealKeys = null;
    if (body.meal_period_key != null) {
      if (Array.isArray(body.meal_period_key)) {
        mealKeys = body.meal_period_key.map(x => String(x).trim()).filter(Boolean);
      } else {
        mealKeys = String(body.meal_period_key).split(',').map(s => s.trim()).filter(Boolean);
      }
      if (!mealKeys.length) mealKeys = null;
    }

    // all -> 不過濾
    const useMealFilter = !!(mealKeys && mealKeys.length && !mealKeys.some(k => k.toLowerCase() === 'all'));

    // 將多個時段轉成「分鐘區間」列表：[{startMin, endMin}, ...]
    let mealRanges = [];
    if (useMealFilter) {
      const mpIndex = new Map(
        mealPeriods
          .filter(p => p && p.enabled !== false && typeof p.key === 'string')
          .map(p => [p.key, p])
      );

      // 驗證 key 都存在
      for (const k of mealKeys) {
        const p = mpIndex.get(k);
        if (!p || !isHHMM(p.start) || !isHHMM(p.end)) {
          return res.status(400).json({ ok: false, error: `invalid meal_period_key: ${k}` });
        }
      }

      // 轉 range
      mealRanges = mealKeys.map(k => {
        const p = mpIndex.get(k);
        const [sh, sm] = p.start.split(':').map(n => parseInt(n, 10));
        const [eh, em] = p.end.split(':').map(n => parseInt(n, 10));
        return { startMin: sh * 60 + sm, endMin: eh * 60 + em };
      });

      // 去重
      const uniq = [];
      const seen = new Set();
      for (const r of mealRanges) {
        const key = `${r.startMin}-${r.endMin}`;
        if (!seen.has(key)) { seen.add(key); uniq.push(r); }
      }
      mealRanges = uniq;
    }

    // -----------------------------
    // store_grading filter: excellent/watch/bad (可多選)
    // -----------------------------
    let gradingFilter = null;
    if (body.store_grading != null) {
      if (Array.isArray(body.store_grading)) {
        gradingFilter = body.store_grading.map(x => String(x).trim()).filter(Boolean);
      } else {
        gradingFilter = String(body.store_grading).split(',').map(s => s.trim()).filter(Boolean);
      }
      if (!gradingFilter.length) gradingFilter = null;

      const allowed = new Set(['excellent', 'watch', 'bad']);
      const badVal = gradingFilter?.find(x => !allowed.has(x));
      if (badVal) {
        return res.status(400).json({ ok: false, error: `invalid store_grading: ${badVal}` });
      }
    }

    // =====================================================
    // aggregation pipeline：productions -> per_store
    // =====================================================
    const pipe = [];

    // time filter (event_time)
    const match = {};
    if (tFrom || tTo) {
      match.event_time = {};
      if (tFrom) match.event_time.$gte = tFrom;
      if (tTo)   match.event_time.$lte = tTo;
    }
    if (Object.keys(match).length) pipe.push({ $match: match });

    // 算三種合格 + ok_all
    pipe.push(
      {
        $project: {
          device_id: 1,
          event_time: 1,
          production_time_sec: 1,
          serving_temp: 1,
          ingredient_missing: 1,

          temp_ok: {
            $and: [
              { $ne: ['$serving_temp', null] },
              { $gte: ['$serving_temp', minServingTemp] }
            ]
          },
          prod_time_ok: {
            $and: [
              { $ne: ['$production_time_sec', null] },
              { $lte: ['$production_time_sec', maxProductionSec] }
            ]
          },
          ingredient_ok: { $ne: ['$ingredient_missing', true] }
        }
      },
      {
        $addFields: {
          ok_all: {
            $cond: [
              { $literal: missingAsNg },
              { $and: ['$temp_ok', '$prod_time_ok', '$ingredient_ok'] },
              { $and: ['$temp_ok', '$prod_time_ok'] }
            ]
          }
        }
      }
    );

    // meal period filter（可複選，多區間 OR）
    if (useMealFilter) {
      pipe.push(
        {
          $addFields: {
            hhmm: { $dateToString: { format: '%H:%M', date: '$event_time', timezone: tz } }
          }
        },
        {
          $addFields: {
            event_min: {
              $add: [
                { $multiply: [{ $toInt: { $substrBytes: ['$hhmm', 0, 2] } }, 60] },
                { $toInt: { $substrBytes: ['$hhmm', 3, 2] } }
              ]
            }
          }
        }
      );

      // 每個 range 自己處理跨午夜，最後用 $or 串起來
      const orConds = [];
      for (const r of mealRanges) {
        if (r.endMin >= r.startMin) {
          orConds.push({ event_min: { $gte: r.startMin, $lt: r.endMin } });
        } else {
          orConds.push({
            $or: [
              { event_min: { $gte: r.startMin } },
              { event_min: { $lt: r.endMin } }
            ]
          });
        }
      }
      pipe.push({ $match: { $or: orConds } });
    }

    // join devices（拿 store + market_id）
    pipe.push(
      {
        $lookup: {
          from: 'devices',
          localField: 'device_id',
          foreignField: 'device_id',
          as: 'dv'
        }
      },
      { $unwind: { path: '$dv', preserveNullAndEmptyArrays: includeUnknown } },

      // ✅ 全部 inclusion，避免投影混用錯誤
      {
        $project: {
          _id: 0,
          temp_ok: 1,
          prod_time_ok: 1,
          ingredient_ok: 1,
          ok_all: 1,

          market_id: includeUnknown ? { $ifNull: ['$dv.store.market_id', 'UNKNOWN'] } : '$dv.store.market_id',
          market: includeUnknown ? { $ifNull: ['$dv.store.market', 'UNKNOWN'] } : '$dv.store.market',
          store_number: includeUnknown ? { $ifNull: ['$dv.store.number', 'UNKNOWN'] } : '$dv.store.number',
          store_name: includeUnknown ? { $ifNull: ['$dv.store.name', 'UNKNOWN'] } : '$dv.store.name',

          latitude: includeUnknown ? { $ifNull: ['$dv.store.latitude', null] } : '$dv.store.latitude',
          longitude: includeUnknown ? { $ifNull: ['$dv.store.longitude', null] } : '$dv.store.longitude'
        }
      }
    );

    // 不含 unknown：排除空店
    if (!includeUnknown) {
      pipe.push({
        $match: {
          market_id: { $ne: null },
          store_number: { $ne: null },
          store_name: { $ne: null }
        }
      });
    }

    // market_id filter
    if (marketIds) {
      pipe.push({ $match: { market_id: { $in: marketIds } } });
    }

    // per_store group
    pipe.push(
      {
        $group: {
          _id: {
            market_id: '$market_id',
            market: '$market',
            number: '$store_number',
            name: '$store_name'
          },
          total: { $sum: 1 },

          temp_ok_count: { $sum: { $cond: ['$temp_ok', 1, 0] } },
          prod_time_ok_count: { $sum: { $cond: ['$prod_time_ok', 1, 0] } },
          ingredient_ok_count: { $sum: { $cond: ['$ingredient_ok', 1, 0] } },
          ok_count: { $sum: { $cond: ['$ok_all', 1, 0] } },

          // latitude: { $first: '$latitude' },
          // longitude:{ $first: '$longitude' }
        }
      },
      {
        $project: {
          _id: 0,
          market_id: '$_id.market_id',
          market: '$_id.market',
          store_number: '$_id.number',
          store_name: '$_id.name',
          // latitude: 1,
          // longitude: 1,
          total: 1,

          temp_ok_ratio: { $round: [{
            $cond: [{ $gt: ['$total', 0] }, { $divide: ['$temp_ok_count', '$total'] }, 0]
          }, 2] },
          prod_time_ok_ratio: { $round: [{
            $cond: [{ $gt: ['$total', 0] }, { $divide: ['$prod_time_ok_count', '$total'] }, 0]
          }, 2] },
          ingredient_ok_ratio: { $round: [{
            $cond: [{ $gt: ['$total', 0] }, { $divide: ['$ingredient_ok_count', '$total'] }, 0]
          }, 2] },
          ok_ratio: { $round: [{
            $cond: [{ $gt: ['$total', 0] }, { $divide: ['$ok_count', '$total'] }, 0]
          }, 2] }


          // temp_ok_ratio: {
          //   $cond: [{ $gt: ['$total', 0] }, { $divide: ['$temp_ok_count', '$total'] }, 0]
          // },
          // prod_time_ok_ratio: {
          //   $cond: [{ $gt: ['$total', 0] }, { $divide: ['$prod_time_ok_count', '$total'] }, 0]
          // },
          // ingredient_ok_ratio: {
          //   $cond: [{ $gt: ['$total', 0] }, { $divide: ['$ingredient_ok_count', '$total'] }, 0]
          // },
          // ok_ratio: {
          //   $cond: [{ $gt: ['$total', 0] }, { $divide: ['$ok_count', '$total'] }, 0]
          // }
        }
      },
      { $sort: { market_id: 1, store_number: 1 } }
    );

    // 執行 aggregation
    let per_store = await col.aggregate(pipe).toArray();

    // 加上 store_grading
    per_store = per_store.map(s => ({
      ...s,
      store_grading: gradeByOkRatio(s.ok_ratio)
    }));

    // store_grading filter（對 per_store 做）
    if (gradingFilter) {
      const set = new Set(gradingFilter);
      per_store = per_store.filter(s => set.has(s.store_grading));
    }

    // =====================================================
    // summary（格式固定照你要求）
    // =====================================================

    // all_stores_number：依 market_id 範圍，devices 裡所有店數
    const storeMatch = {};
    if (marketIds) storeMatch['store.market_id'] = { $in: marketIds };

    // ✅ 這裡統一用 deviceCol（你原本寫 devCol 容易炸）
    const allStores = await devCol.distinct('store.number', storeMatch);
    const all_stores_number = allStores.length;

    // total_stores：過濾後 per_store 的店數
    const total_stores = per_store.length;

    // total_productions：過濾後 production 總筆數
    const total_productions = per_store.reduce((sum, s) => sum + (s.total || 0), 0);

    // period_store_category：用過濾後 per_store 的分級統計
    let excellent_count = 0, watch_count = 0, bad_count = 0;
    for (const s of per_store) {
      if (s.store_grading === 'excellent') excellent_count++;
      else if (s.store_grading === 'watch') watch_count++;
      else bad_count++;
    }

    const period_store_category = {
      excellent_count,
      watch_count,
      bad_count,
      total_stores,

      excellent_ratio: Math.round((total_stores ? (excellent_count / total_stores) : 0) * 100)/100,
      watch_ratio: Math.round((total_stores ? (watch_count / total_stores) : 0) * 100)/100,
      bad_ratio: Math.round((total_stores ? (bad_count / total_stores) : 0) * 100)/100

      // excellent_ratio: total_stores ? excellent_count / total_stores : 0,
      // watch_ratio: total_stores ? watch_count / total_stores : 0,
      // bad_ratio: total_stores ? bad_count / total_stores : 0
    };

    const summary = {
      all_stores_number,
      total_stores,
      store_number_string: `${total_stores} / ${all_stores_number}`,
      total_productions,
      period_store_category
    };

    // 回傳
    return res.json({
      ok: true,
      data: { per_store },
      summary
    });

  } catch (err) {
    console.error('[POST /store-ok-stats] error:', err);
    return res.status(500).json({ ok: false, error: String(err?.message || err) });
  }
});
// app.post('/store-ok-stats', async (req, res) => {
//   try {
//     const body = req.body || {};

//     // -----------------------------
//     // 基本參數
//     // -----------------------------
//     const tFrom = parseISO(body.from);
//     const tTo   = parseISO(body.to);
//     const tz    = String(body.tz || 'UTC');
//     const includeUnknown = ['1', 'true', 'True'].includes(String(body.include_unknown || '0'));

//     // market_id filter: string | array | csv
//     let marketIds = null;
//     if (body.market_id != null) {
//       if (Array.isArray(body.market_id)) {
//         marketIds = body.market_id.map(x => String(x).trim()).filter(Boolean);
//       } else {
//         marketIds = String(body.market_id).split(',').map(s => s.trim()).filter(Boolean);
//       }
//       if (!marketIds.length) marketIds = null;
//     }

//     // -----------------------------
//     // 讀 settings：quality_settings（完全依你貼的 qsDoc）
//     // -----------------------------
//     const qsDoc = await settingCol.findOne({ type: 'quality_settings' });
//     if (!qsDoc) {
//       return res.status(500).json({ ok: false, error: 'quality_settings not found in settings collection' });
//     }

//     // ramen_standard 門檻
//     const minServingTemp = Number(qsDoc?.ramen_standard?.min_temp_c);
//     const maxProductionSec = Number(qsDoc?.ramen_standard?.max_production_time_sec);
//     if (!Number.isFinite(minServingTemp) || !Number.isFinite(maxProductionSec)) {
//       return res.status(500).json({
//         ok: false,
//         error: 'quality_settings.ramen_standard invalid (min_temp_c / max_production_time_sec)'
//       });
//     }

//     // 是否把缺料當 NG：你目前 qsDoc 沒這欄，先固定 true（最符合「食材完整」概念）
//     const missingAsNg = true;

//     // store_grading 門檻
//     const excellentMin = Number(qsDoc?.store_grading?.excellent_ratio_min);
//     const watchMin = Number(qsDoc?.store_grading?.watch_ratio_min);
//     const badMax = Number(qsDoc?.store_grading?.bad_ratio_max);

//     if (!Number.isFinite(excellentMin) || !Number.isFinite(watchMin)) {
//       return res.status(500).json({
//         ok: false,
//         error: 'quality_settings.store_grading invalid (excellent_ratio_min / watch_ratio_min)'
//       });
//     }

//     // 一致性提醒（不影響執行）
//     if (Number.isFinite(badMax) && Math.abs(badMax - watchMin) > 1e-9) {
//       console.warn('[quality_settings] bad_ratio_max != watch_ratio_min, using watch_ratio_min as boundary');
//     }

//     function gradeByOkRatio(okRatio) {
//       const r = Number.isFinite(Number(okRatio)) ? Number(okRatio) : 0;
//       if (r >= excellentMin) return 'excellent';
//       if (r >= watchMin) return 'watch';
//       return 'bad';
//     }

//     // meal periods（完全用 qsDoc.meal_periods）
//     const mealPeriods = Array.isArray(qsDoc?.meal_periods) ? qsDoc.meal_periods : [];

//     // -----------------------------
//     // meal_period_key filter
//     // -----------------------------
//     const mealKey = body.meal_period_key ? String(body.meal_period_key).trim() : null;
//     const useMealFilter = mealKey && mealKey.toLowerCase() !== 'all';

//     let mealStartMin = null;
//     let mealEndMin = null;
//     if (useMealFilter) {
//       const p = mealPeriods.find(x => x && x.key === mealKey && x.enabled !== false);
//       if (!p || !isHHMM(p.start) || !isHHMM(p.end)) {
//         return res.status(400).json({ ok: false, error: `invalid meal_period_key: ${mealKey}` });
//       }
//       const [sh, sm] = p.start.split(':').map(n => parseInt(n, 10));
//       const [eh, em] = p.end.split(':').map(n => parseInt(n, 10));
//       mealStartMin = sh * 60 + sm;
//       mealEndMin   = eh * 60 + em;
//     }

//     // -----------------------------
//     // store_grading filter: excellent/watch/bad (可多選)
//     // -----------------------------
//     let gradingFilter = null;
//     if (body.store_grading != null) {
//       if (Array.isArray(body.store_grading)) {
//         gradingFilter = body.store_grading.map(x => String(x).trim()).filter(Boolean);
//       } else {
//         gradingFilter = String(body.store_grading).split(',').map(s => s.trim()).filter(Boolean);
//       }
//       if (!gradingFilter.length) gradingFilter = null;

//       const allowed = new Set(['excellent', 'watch', 'bad']);
//       const badVal = gradingFilter?.find(x => !allowed.has(x));
//       if (badVal) {
//         return res.status(400).json({ ok: false, error: `invalid store_grading: ${badVal}` });
//       }
//     }

//     // =====================================================
//     // aggregation pipeline：productions -> per_store
//     // =====================================================
//     const pipe = [];

//     // time filter (event_time)
//     const match = {};
//     if (tFrom || tTo) {
//       match.event_time = {};
//       if (tFrom) match.event_time.$gte = tFrom;
//       if (tTo)   match.event_time.$lte = tTo;
//     }
//     if (Object.keys(match).length) pipe.push({ $match: match });

//     // 算三種合格 + ok_all
//     pipe.push(
//       {
//         $project: {
//           device_id: 1,
//           event_time: 1,
//           production_time_sec: 1,
//           serving_temp: 1,
//           ingredient_missing: 1,

//           temp_ok: {
//             $and: [
//               { $ne: ['$serving_temp', null] },
//               { $gte: ['$serving_temp', minServingTemp] }
//             ]
//           },
//           prod_time_ok: {
//             $and: [
//               { $ne: ['$production_time_sec', null] },
//               { $lte: ['$production_time_sec', maxProductionSec] }
//             ]
//           },
//           ingredient_ok: { $ne: ['$ingredient_missing', true] }
//         }
//       },
//       {
//         $addFields: {
//           ok_all: {
//             $cond: [
//               { $literal: missingAsNg },
//               { $and: ['$temp_ok', '$prod_time_ok', '$ingredient_ok'] },
//               { $and: ['$temp_ok', '$prod_time_ok'] }
//             ]
//           }
//         }
//       }
//     );

//     // meal period filter（用 tz 把 event_time 切成當地 HH:MM）
//     if (useMealFilter) {
//       pipe.push(
//         {
//           $addFields: {
//             hhmm: { $dateToString: { format: '%H:%M', date: '$event_time', timezone: tz } }
//           }
//         },
//         {
//           $addFields: {
//             event_min: {
//               $add: [
//                 { $multiply: [{ $toInt: { $substrBytes: ['$hhmm', 0, 2] } }, 60] },
//                 { $toInt: { $substrBytes: ['$hhmm', 3, 2] } }
//               ]
//             }
//           }
//         }
//       );

//       // 跨午夜支援（雖然你目前時段不跨，但先補）
//       if (mealEndMin >= mealStartMin) {
//         pipe.push({ $match: { event_min: { $gte: mealStartMin, $lt: mealEndMin } } });
//       } else {
//         pipe.push({
//           $match: {
//             $or: [
//               { event_min: { $gte: mealStartMin } },
//               { event_min: { $lt: mealEndMin } }
//             ]
//           }
//         });
//       }
//     }

//     // join devices（拿 store + market_id）
//     pipe.push(
//       {
//         $lookup: {
//           from: 'devices',
//           localField: 'device_id',
//           foreignField: 'device_id',
//           as: 'dv'
//         }
//       },
//       { $unwind: { path: '$dv', preserveNullAndEmptyArrays: includeUnknown } },

//       // ✅ 全部 inclusion，避免投影混用錯誤
//       {
//         $project: {
//           _id: 0,
//           temp_ok: 1,
//           prod_time_ok: 1,
//           ingredient_ok: 1,
//           ok_all: 1,

//           market_id: includeUnknown ? { $ifNull: ['$dv.store.market_id', 'UNKNOWN'] } : '$dv.store.market_id',
//           market: includeUnknown ? { $ifNull: ['$dv.store.market', 'UNKNOWN'] } : '$dv.store.market',
//           store_number: includeUnknown ? { $ifNull: ['$dv.store.number', 'UNKNOWN'] } : '$dv.store.number',
//           store_name: includeUnknown ? { $ifNull: ['$dv.store.name', 'UNKNOWN'] } : '$dv.store.name',

//           latitude: includeUnknown ? { $ifNull: ['$dv.store.latitude', null] } : '$dv.store.latitude',
//           longitude: includeUnknown ? { $ifNull: ['$dv.store.longitude', null] } : '$dv.store.longitude'
//         }
//       }
//     );

//     // 不含 unknown：排除空店
//     if (!includeUnknown) {
//       pipe.push({
//         $match: {
//           market_id: { $ne: null },
//           store_number: { $ne: null },
//           store_name: { $ne: null }
//         }
//       });
//     }

//     // market_id filter
//     if (marketIds) {
//       pipe.push({ $match: { market_id: { $in: marketIds } } });
//     }

//     // per_store group
//     pipe.push(
//       {
//         $group: {
//           _id: {
//             market_id: '$market_id',
//             market: '$market',
//             number: '$store_number',
//             name: '$store_name'
//           },
//           total: { $sum: 1 },

//           temp_ok_count: { $sum: { $cond: ['$temp_ok', 1, 0] } },
//           prod_time_ok_count: { $sum: { $cond: ['$prod_time_ok', 1, 0] } },
//           ingredient_ok_count: { $sum: { $cond: ['$ingredient_ok', 1, 0] } },
//           ok_count: { $sum: { $cond: ['$ok_all', 1, 0] } },

//           latitude: { $first: '$latitude' },
//           longitude:{ $first: '$longitude' }
//         }
//       },
//       {
//         $project: {
//           _id: 0,
//           market_id: '$_id.market_id',
//           market: '$_id.market',
//           store_number: '$_id.number',
//           store_name: '$_id.name',
//           latitude: 1,
//           longitude: 1,
//           total: 1,

//           temp_ok_ratio: {
//             $cond: [{ $gt: ['$total', 0] }, { $divide: ['$temp_ok_count', '$total'] }, 0]
//           },
//           prod_time_ok_ratio: {
//             $cond: [{ $gt: ['$total', 0] }, { $divide: ['$prod_time_ok_count', '$total'] }, 0]
//           },
//           ingredient_ok_ratio: {
//             $cond: [{ $gt: ['$total', 0] }, { $divide: ['$ingredient_ok_count', '$total'] }, 0]
//           },
//           ok_ratio: {
//             $cond: [{ $gt: ['$total', 0] }, { $divide: ['$ok_count', '$total'] }, 0]
//           }
//         }
//       },
//       { $sort: { market_id: 1, store_number: 1 } }
//     );

//     // 執行 aggregation
//     let per_store = await col.aggregate(pipe).toArray();

//     // 加上 store_grading
//     per_store = per_store.map(s => ({
//       ...s,
//       store_grading: gradeByOkRatio(s.ok_ratio)
//     }));

//     // store_grading filter（對 per_store 做）
//     if (gradingFilter) {
//       const set = new Set(gradingFilter);
//       per_store = per_store.filter(s => set.has(s.store_grading));
//     }

//     // =====================================================
//     // summary（格式固定照你要求）
//     // =====================================================

//     // all_stores_number：依 market_id 範圍，devices 裡所有店數
//     const storeMatch = {};
//     if (marketIds) storeMatch['store.market_id'] = { $in: marketIds };
//     const allStores = await devCol.distinct('store.number', storeMatch);
//     const all_stores_number = allStores.length;

//     // total_stores：過濾後 per_store 的店數
//     const total_stores = per_store.length;

//     // total_productions：過濾後 production 總筆數
//     const total_productions = per_store.reduce((sum, s) => sum + (s.total || 0), 0);

//     // period_store_category：用過濾後 per_store 的分級統計
//     let excellent_count = 0, watch_count = 0, bad_count = 0;
//     for (const s of per_store) {
//       if (s.store_grading === 'excellent') excellent_count++;
//       else if (s.store_grading === 'watch') watch_count++;
//       else bad_count++;
//     }

//     const period_store_category = {
//       excellent_count,
//       watch_count,
//       bad_count,
//       total_stores,
//       excellent_ratio: total_stores ? excellent_count / total_stores : 0,
//       watch_ratio: total_stores ? watch_count / total_stores : 0,
//       bad_ratio: total_stores ? bad_count / total_stores : 0
//     };

//     const summary = {
//       all_stores_number,
//       total_stores,
//       store_number_string: `${total_stores} / ${all_stores_number}`,
//       total_productions,
//       period_store_category
//     };

//     // 回傳
//     return res.json({
//       ok: true,
//       data: { per_store }, // ✅ 不含 daily_store_category
//       summary
//     });

//   } catch (err) {
//     console.error('[POST /quality/store-ok-stats] error:', err);
//     return res.status(500).json({ ok: false, error: String(err?.message || err) });
//   }
// });


// =====================================================
// POST /quality/meal-period-stats
// 依餐期統計：製作數 / 溫度合格率 / 配麵時間合格率 / 食材完整性合格率
// 過濾：market_id、store_number(可複選/all)、時間區間、include_unknown
// =====================================================
app.post('/meal-period-stats', async (req, res) => {
  try {
    const body = req.body || {};

    const tFrom = parseISO(body.from);
    const tTo   = parseISO(body.to);
    const tz    = String(body.tz || 'Asia/Shanghai');
    const includeUnknown = ['1', 'true', 'True'].includes(String(body.include_unknown || '0'));

    // market_id filter
    let marketIds = null;
    if (body.market_id != null) {
      if (Array.isArray(body.market_id)) {
        marketIds = body.market_id.map(x => String(x).trim()).filter(Boolean);
      } else {
        marketIds = String(body.market_id).split(',').map(s => s.trim()).filter(Boolean);
      }
      if (!marketIds.length) marketIds = null;
    }

    // store_number filter (可複選/all)
    let storeNumbers = null;
    if (body.store_number != null) {
      if (Array.isArray(body.store_number)) {
        storeNumbers = body.store_number.map(x => String(x).trim()).filter(Boolean);
      } else {
        storeNumbers = String(body.store_number).split(',').map(s => s.trim()).filter(Boolean);
      }
      if (!storeNumbers.length) storeNumbers = null;
      if (storeNumbers && storeNumbers.some(x => x.toLowerCase() === 'all')) storeNumbers = null;
    }

    // ✅ meal_period_key filter (可複選/all)
    let mealKeys = null;
    if (body.meal_period_key != null) {
      if (Array.isArray(body.meal_period_key)) {
        mealKeys = body.meal_period_key.map(x => String(x).trim()).filter(Boolean);
      } else {
        mealKeys = String(body.meal_period_key).split(',').map(s => s.trim()).filter(Boolean);
      }
      if (!mealKeys.length) mealKeys = null;
      if (mealKeys && mealKeys.some(x => x.toLowerCase() === 'all')) mealKeys = null;
    }

    // 讀 settings：quality_settings
    const qsDoc = await settingCol.findOne({ type: 'quality_settings' });
    if (!qsDoc) {
      return res.status(500).json({ ok: false, error: 'quality_settings not found in settings collection' });
    }

    const minServingTemp = Number(qsDoc?.ramen_standard?.min_temp_c);
    const maxProductionSec = Number(qsDoc?.ramen_standard?.max_production_time_sec);
    if (!Number.isFinite(minServingTemp) || !Number.isFinite(maxProductionSec)) {
      return res.status(500).json({
        ok: false,
        error: 'quality_settings.ramen_standard invalid (min_temp_c / max_production_time_sec)'
      });
    }

    const mealPeriods = Array.isArray(qsDoc?.meal_periods) ? qsDoc.meal_periods.filter(p => p?.enabled !== false) : [];
    if (!mealPeriods.length) {
      return res.status(500).json({ ok: false, error: 'quality_settings.meal_periods is empty' });
    }

    // 驗證 mealPeriods 格式
    for (const p of mealPeriods) {
      if (!p.key || !isHHMM(p.start) || !isHHMM(p.end)) {
        return res.status(500).json({ ok: false, error: `invalid meal_periods item: ${JSON.stringify(p)}` });
      }
    }

    // ✅ 驗證 meal_period_key（若有指定）
    const mealKeySet = new Set(mealPeriods.map(p => p.key));
    if (mealKeys) {
      for (const k of mealKeys) {
        if (!mealKeySet.has(k)) {
          return res.status(400).json({ ok: false, error: `invalid meal_period_key: ${k}` });
        }
      }
    }

    // -----------------------------------------------------
    // 動態產生 $switch branches：把 event_min 分配到 period_key
    // -----------------------------------------------------
    const switchBranches = mealPeriods.map(p => {
      const startMin = hhmmToMin(p.start);
      const endMin   = hhmmToMin(p.end);

      // 跨午夜支援
      const cond = (endMin >= startMin)
        ? { $and: [{ $gte: ['$event_min', startMin] }, { $lt: ['$event_min', endMin] }] }
        : { $or:  [{ $gte: ['$event_min', startMin] }, { $lt: ['$event_min', endMin] }] };

      return { case: cond, then: p.key };
    });

    // -----------------------------------------------------
    // Pipeline
    // -----------------------------------------------------
    const pipe = [];

    // time filter by event_time
    const match = {};
    if (tFrom || tTo) {
      match.event_time = {};
      if (tFrom) match.event_time.$gte = tFrom;
      if (tTo)   match.event_time.$lte = tTo;
    }
    if (Object.keys(match).length) pipe.push({ $match: match });

    // 計算三種合格
    pipe.push(
      {
        $project: {
          device_id: 1,
          event_time: 1,
          production_time_sec: 1,
          serving_temp: 1,
          ingredient_missing: 1,

          temp_ok: {
            $and: [
              { $ne: ['$serving_temp', null] },
              { $gte: ['$serving_temp', minServingTemp] }
            ]
          },
          prod_time_ok: {
            $and: [
              { $ne: ['$production_time_sec', null] },
              { $lte: ['$production_time_sec', maxProductionSec] }
            ]
          },
          ingredient_ok: { $ne: ['$ingredient_missing', true] }
        }
      },
      // event_min：以 tz 轉成 HH:MM 再算分鐘
      {
        $addFields: {
          hhmm: { $dateToString: { format: '%H:%M', date: '$event_time', timezone: tz } }
        }
      },
      {
        $addFields: {
          event_min: {
            $add: [
              { $multiply: [{ $toInt: { $substrBytes: ['$hhmm', 0, 2] } }, 60] },
              { $toInt: { $substrBytes: ['$hhmm', 3, 2] } }
            ]
          }
        }
      },
      // period_key：依 mealPeriods 分配
      {
        $addFields: {
          period_key: {
            $switch: {
              branches: switchBranches,
              default: 'UNKNOWN_PERIOD'
            }
          }
        }
      }
    );

    // join devices（用 store.market_id / store.number 過濾）
    pipe.push(
      {
        $lookup: {
          from: 'devices',
          localField: 'device_id',
          foreignField: 'device_id',
          as: 'dv'
        }
      },
      { $unwind: { path: '$dv', preserveNullAndEmptyArrays: includeUnknown } },
      {
        $project: {
          _id: 0,
          period_key: 1,
          temp_ok: 1,
          prod_time_ok: 1,
          ingredient_ok: 1,

          market_id: includeUnknown ? { $ifNull: ['$dv.store.market_id', 'UNKNOWN'] } : '$dv.store.market_id',
          store_number: includeUnknown ? { $ifNull: ['$dv.store.number', 'UNKNOWN'] } : '$dv.store.number'
        }
      }
    );

    if (!includeUnknown) {
      pipe.push({
        $match: {
          market_id: { $ne: null },
          store_number: { $ne: null }
        }
      });
    }

    if (marketIds) {
      pipe.push({ $match: { market_id: { $in: marketIds } } });
    }
    if (storeNumbers) {
      pipe.push({ $match: { store_number: { $in: storeNumbers } } });
    }

    // 只保留有效餐期（把 UNKNOWN_PERIOD 擋掉）
    pipe.push({ $match: { period_key: { $ne: 'UNKNOWN_PERIOD' } } });

    // ✅ 新增：餐期過濾（多選）
    if (mealKeys) {
      pipe.push({ $match: { period_key: { $in: mealKeys } } });
    }

    // 依餐期 group
    pipe.push(
      {
        $group: {
          _id: '$period_key',
          total_productions: { $sum: 1 },
          temp_ok_count: { $sum: { $cond: ['$temp_ok', 1, 0] } },
          prod_time_ok_count: { $sum: { $cond: ['$prod_time_ok', 1, 0] } },
          ingredient_ok_count: { $sum: { $cond: ['$ingredient_ok', 1, 0] } }
        }
      },
      {
        $project: {
          _id: 0,
          key: '$_id',
          total_productions: 1,
          temp_ok_ratio: { $round: [{
            $cond: [{ $gt: ['$total_productions', 0] }, { $divide: ['$temp_ok_count', '$total_productions'] }, 0]
          }, 2] },
          prod_time_ok_ratio: { $round: [{
            $cond: [{ $gt: ['$total_productions', 0] }, { $divide: ['$prod_time_ok_count', '$total_productions'] }, 0]
          }, 2] },
          ingredient_ok_ratio: { $round: [{
            $cond: [{ $gt: ['$total_productions', 0] }, { $divide: ['$ingredient_ok_count', '$total_productions'] }, 0]
          }, 2] },
          // temp_ok_ratio: {
          //   $cond: [{ $gt: ['$total_productions', 0] }, { $divide: ['$temp_ok_count', '$total_productions'] }, 0]
          // },
          // prod_time_ok_ratio: {
          //   $cond: [{ $gt: ['$total_productions', 0] }, { $divide: ['$prod_time_ok_count', '$total_productions'] }, 0]
          // },
          // ingredient_ok_ratio: {
          //   $cond: [{ $gt: ['$total_productions', 0] }, { $divide: ['$ingredient_ok_count', '$total_productions'] }, 0]
          // }
        }
      }
    );

    const rows = await col.aggregate(pipe).toArray();

    // -----------------------------------------------------
    // 補 0 並依 settings 順序輸出（✅ 若有 mealKeys，就只輸出那些 key）
    // -----------------------------------------------------
    const byKey = new Map(rows.map(r => [r.key, r]));

    const targetPeriods = mealKeys
      ? mealPeriods.filter(p => mealKeys.includes(p.key))
      : mealPeriods;

    const periods = targetPeriods.map(p => {
      const r = byKey.get(p.key) || {
        key: p.key,
        total_productions: 0,
        temp_ok_ratio: 1,         // 0 筆時顯示 100%（你原本邏輯）
        prod_time_ok_ratio: 1,
        ingredient_ok_ratio: 1
      };
      return {
        key: p.key,
        name: p.name,
        start: p.start,
        end: p.end,
        total_productions: r.total_productions,
        temp_ok_ratio: r.temp_ok_ratio,
        prod_time_ok_ratio: r.prod_time_ok_ratio,
        ingredient_ok_ratio: r.ingredient_ok_ratio
      };
    });

    // summary（匯總）
    const total_productions = periods.reduce((s, x) => s + (x.total_productions || 0), 0);

    // 匯總 ratio 用加權：sum(ok_count)/sum(total)
    const sumTempOk = periods.reduce((s, x) => s + (x.total_productions * x.temp_ok_ratio), 0);
    const sumProdOk = periods.reduce((s, x) => s + (x.total_productions * x.prod_time_ok_ratio), 0);
    const sumIngOk  = periods.reduce((s, x) => s + (x.total_productions * x.ingredient_ok_ratio), 0);

    // all_stores_number：依 market_id 範圍，devices 裡所有店數
    const storeMatch = {};
    if (marketIds) storeMatch['store.market_id'] = { $in: marketIds };
    const allStores = await devCol.distinct('store.number', storeMatch);
    const all_stores_number = allStores.length ? (allStores.length) : 0;
    const stores_number = storeNumbers != null ? (storeNumbers.length) : all_stores_number;

    const summary = {
      total_productions,
      temp_ok_ratio: Math.round((total_productions ? (sumTempOk / total_productions) : 0) * 100)/100,
      prod_time_ok_ratio: Math.round((total_productions ? (sumProdOk / total_productions) : 0) * 100)/100,
      ingredient_ok_ratio: Math.round((total_productions ? (sumIngOk / total_productions) : 0) * 100)/100,

      // temp_ok_ratio: total_productions ? (sumTempOk / total_productions) : 0,
      // prod_time_ok_ratio: total_productions ? (sumProdOk / total_productions) : 0,
      // ingredient_ok_ratio: total_productions ? (sumIngOk / total_productions) : 0,
      all_stores_number: all_stores_number,
      stores_number: stores_number
    };
  
    return res.json({
      ok: true,
      data: { summary, periods }
    });

  } catch (err) {
    console.error('[POST /quality/meal-period-stats] error:', err);
    return res.status(500).json({ ok: false, error: String(err?.message || err) });
  }
});

// app.post('/meal-period-stats', async (req, res) => {
//   try {
//     const body = req.body || {};

//     const tFrom = parseISO(body.from);
//     const tTo   = parseISO(body.to);
//     const tz    = String(body.tz || 'Asia/Shanghai');
//     const includeUnknown = ['1', 'true', 'True'].includes(String(body.include_unknown || '0'));

//     // market_id filter
//     let marketIds = null;
//     if (body.market_id != null) {
//       if (Array.isArray(body.market_id)) {
//         marketIds = body.market_id.map(x => String(x).trim()).filter(Boolean);
//       } else {
//         marketIds = String(body.market_id).split(',').map(s => s.trim()).filter(Boolean);
//       }
//       if (!marketIds.length) marketIds = null;
//     }

//     // store_number filter (可複選/all)
//     let storeNumbers = null;
//     if (body.store_number != null) {
//       if (Array.isArray(body.store_number)) {
//         storeNumbers = body.store_number.map(x => String(x).trim()).filter(Boolean);
//       } else {
//         storeNumbers = String(body.store_number).split(',').map(s => s.trim()).filter(Boolean);
//       }
//       if (!storeNumbers.length) storeNumbers = null;
//       if (storeNumbers && storeNumbers.some(x => x.toLowerCase() === 'all')) storeNumbers = null;
//     }

//     // 讀 settings：quality_settings（依你提供的 qsDoc）
//     const qsDoc = await settingCol.findOne({ type: 'quality_settings' });
//     if (!qsDoc) {
//       return res.status(500).json({ ok: false, error: 'quality_settings not found in settings collection' });
//     }

//     const minServingTemp = Number(qsDoc?.ramen_standard?.min_temp_c);
//     const maxProductionSec = Number(qsDoc?.ramen_standard?.max_production_time_sec);
//     if (!Number.isFinite(minServingTemp) || !Number.isFinite(maxProductionSec)) {
//       return res.status(500).json({
//         ok: false,
//         error: 'quality_settings.ramen_standard invalid (min_temp_c / max_production_time_sec)'
//       });
//     }

//     const mealPeriods = Array.isArray(qsDoc?.meal_periods) ? qsDoc.meal_periods.filter(p => p?.enabled !== false) : [];
//     if (!mealPeriods.length) {
//       return res.status(500).json({ ok: false, error: 'quality_settings.meal_periods is empty' });
//     }

//     // 驗證 mealPeriods 格式（避免有人把 start/end 寫爛）
//     for (const p of mealPeriods) {
//       if (!p.key || !isHHMM(p.start) || !isHHMM(p.end)) {
//         return res.status(500).json({ ok: false, error: `invalid meal_periods item: ${JSON.stringify(p)}` });
//       }
//     }

//     // -----------------------------------------------------
//     // 動態產生 $switch branches：把 event_min 分配到 period_key
//     // -----------------------------------------------------
//     const switchBranches = mealPeriods.map(p => {
//       const startMin = hhmmToMin(p.start);
//       const endMin   = hhmmToMin(p.end);

//       // 跨午夜支援
//       const cond = (endMin >= startMin)
//         ? { $and: [{ $gte: ['$event_min', startMin] }, { $lt: ['$event_min', endMin] }] }
//         : { $or:  [{ $gte: ['$event_min', startMin] }, { $lt: ['$event_min', endMin] }] };

//       return { case: cond, then: p.key };
//     });

//     // -----------------------------------------------------
//     // Pipeline
//     // -----------------------------------------------------
//     const pipe = [];

//     // time filter by event_time
//     const match = {};
//     if (tFrom || tTo) {
//       match.event_time = {};
//       if (tFrom) match.event_time.$gte = tFrom;
//       if (tTo)   match.event_time.$lte = tTo;
//     }
//     if (Object.keys(match).length) pipe.push({ $match: match });

//     // 計算三種合格（注意：ingredient_missing == true 視為不完整）
//     pipe.push(
//       {
//         $project: {
//           device_id: 1,
//           event_time: 1,
//           production_time_sec: 1,
//           serving_temp: 1,
//           ingredient_missing: 1,

//           temp_ok: {
//             $and: [
//               { $ne: ['$serving_temp', null] },
//               { $gte: ['$serving_temp', minServingTemp] }
//             ]
//           },
//           prod_time_ok: {
//             $and: [
//               { $ne: ['$production_time_sec', null] },
//               { $lte: ['$production_time_sec', maxProductionSec] }
//             ]
//           },
//           ingredient_ok: { $ne: ['$ingredient_missing', true] }
//         }
//       },
//       // event_min：以 tz 轉成 HH:MM 再算分鐘
//       {
//         $addFields: {
//           hhmm: { $dateToString: { format: '%H:%M', date: '$event_time', timezone: tz } }
//         }
//       },
//       {
//         $addFields: {
//           event_min: {
//             $add: [
//               { $multiply: [{ $toInt: { $substrBytes: ['$hhmm', 0, 2] } }, 60] },
//               { $toInt: { $substrBytes: ['$hhmm', 3, 2] } }
//             ]
//           }
//         }
//       },
//       // period_key：依 mealPeriods 分配（找不到的標成 UNKNOWN_PERIOD）
//       {
//         $addFields: {
//           period_key: {
//             $switch: {
//               branches: switchBranches,
//               default: 'UNKNOWN_PERIOD'
//             }
//           }
//         }
//       }
//     );

//     // join devices（用 store.market_id / store.number 過濾）
//     pipe.push(
//       {
//         $lookup: {
//           from: 'devices',
//           localField: 'device_id',
//           foreignField: 'device_id',
//           as: 'dv'
//         }
//       },
//       { $unwind: { path: '$dv', preserveNullAndEmptyArrays: includeUnknown } },
//       {
//         $project: {
//           _id: 0,
//           period_key: 1,
//           temp_ok: 1,
//           prod_time_ok: 1,
//           ingredient_ok: 1,

//           market_id: includeUnknown ? { $ifNull: ['$dv.store.market_id', 'UNKNOWN'] } : '$dv.store.market_id',
//           store_number: includeUnknown ? { $ifNull: ['$dv.store.number', 'UNKNOWN'] } : '$dv.store.number'
//         }
//       }
//     );

//     if (!includeUnknown) {
//       pipe.push({
//         $match: {
//           market_id: { $ne: null },
//           store_number: { $ne: null }
//         }
//       });
//     }

//     if (marketIds) {
//       pipe.push({ $match: { market_id: { $in: marketIds } } });
//     }
//     if (storeNumbers) {
//       pipe.push({ $match: { store_number: { $in: storeNumbers } } });
//     }

//     // 只保留有效餐期（把 UNKNOWN_PERIOD 擋掉，避免污染）
//     pipe.push({ $match: { period_key: { $ne: 'UNKNOWN_PERIOD' } } });

//     // 依餐期 group
//     pipe.push(
//       {
//         $group: {
//           _id: '$period_key',
//           total_productions: { $sum: 1 },
//           temp_ok_count: { $sum: { $cond: ['$temp_ok', 1, 0] } },
//           prod_time_ok_count: { $sum: { $cond: ['$prod_time_ok', 1, 0] } },
//           ingredient_ok_count: { $sum: { $cond: ['$ingredient_ok', 1, 0] } }
//         }
//       },
//       {
//         $project: {
//           _id: 0,
//           key: '$_id',
//           total_productions: 1,
//           temp_ok_ratio: {
//             $cond: [{ $gt: ['$total_productions', 0] }, { $divide: ['$temp_ok_count', '$total_productions'] }, 0]
//           },
//           prod_time_ok_ratio: {
//             $cond: [{ $gt: ['$total_productions', 0] }, { $divide: ['$prod_time_ok_count', '$total_productions'] }, 0]
//           },
//           ingredient_ok_ratio: {
//             $cond: [{ $gt: ['$total_productions', 0] }, { $divide: ['$ingredient_ok_count', '$total_productions'] }, 0]
//           }
//         }
//       }
//     );

//     const rows = await col.aggregate(pipe).toArray();

//     // -----------------------------------------------------
//     // 把缺的餐期補 0，並依 settings 順序輸出（對 UI 很重要）
//     // -----------------------------------------------------
//     const byKey = new Map(rows.map(r => [r.key, r]));

//     const periods = mealPeriods.map(p => {
//       const r = byKey.get(p.key) || {
//         key: p.key,
//         total_productions: 0,
//         temp_ok_ratio: 1,         // 0 筆時顯示 100% 比較符合你畫面（可改 0）
//         prod_time_ok_ratio: 1,
//         ingredient_ok_ratio: 1
//       };
//       return {
//         key: p.key,
//         name: p.name,
//         start: p.start,
//         end: p.end,
//         total_productions: r.total_productions,
//         temp_ok_ratio: r.temp_ok_ratio,
//         prod_time_ok_ratio: r.prod_time_ok_ratio,
//         ingredient_ok_ratio: r.ingredient_ok_ratio
//       };
//     });

//     // summary（匯總）
//     const total_productions = periods.reduce((s, x) => s + (x.total_productions || 0), 0);

//     // 匯總 ratio 用「加權」：sum(ok_count)/sum(total)
//     const sumTempOk = periods.reduce((s, x) => s + (x.total_productions * x.temp_ok_ratio), 0);
//     const sumProdOk = periods.reduce((s, x) => s + (x.total_productions * x.prod_time_ok_ratio), 0);
//     const sumIngOk  = periods.reduce((s, x) => s + (x.total_productions * x.ingredient_ok_ratio), 0);

//     // all_stores_number：依 market_id 範圍，devices 裡所有店數
//     const storeMatch = {};
//     if (marketIds) storeMatch['store.market_id'] = { $in: marketIds };
//     const allStores = await devCol.distinct('store.number', storeMatch);
//     const all_stores_number = allStores.length ? (allStores.length) : 0
//     const stores_number = storeNumbers != null ? (storeNumbers.length) : all_stores_number

//     const summary = {
//       total_productions,
//       temp_ok_ratio: total_productions ? (sumTempOk / total_productions) : 0,
//       prod_time_ok_ratio: total_productions ? (sumProdOk / total_productions) : 0,
//       ingredient_ok_ratio: total_productions ? (sumIngOk / total_productions) : 0,
//       all_stores_number: all_stores_number,
//       stores_number: stores_number
//     };

//     return res.json({
//       ok: true,
//       data: { summary, periods }
//     });

//   } catch (err) {
//     console.error('[POST /quality/meal-period-stats] error:', err);
//     return res.status(500).json({ ok: false, error: String(err?.message || err) });
//   }
// });


// =====================================================
// POST /quality/entries
// - filters: meal_period_key (multi), store_number (multi), store_grading (multi)
// - output: add ramen_name
// =====================================================
app.post('/entries', async (req, res) => {
  try {
    const body = req.body || {};

    const includeRaw = ['1','true','True'].includes(String(body.include_raw || '0'));
    const includeUnknown = ['1','true','True'].includes(String(body.include_unknown || '0'));

    const tFrom = parseISO(body.from);
    const tTo   = parseISO(body.to);
    const tz    = String(body.tz || 'Asia/Shanghai');

    // ✅ sort
    const sortBy = String(body.sort_by || 'event_time_asc');

    // filters (multi)
    const marketIds    = parseMulti(body.market_id);
    const storeNumbers = parseMulti(body.store_number, { allowAll: true });
    const mealKeys     = parseMulti(body.meal_period_key, { allowAll: true });
    const deviceIds    = parseMulti(body.device_id);

    let gradingFilter = parseMulti(body.store_grading);
    if (gradingFilter) {
      const allowed = new Set(['excellent','watch','bad']);
      const badVal = gradingFilter.find(x => !allowed.has(x));
      if (badVal) return res.status(400).json({ ok: false, error: `invalid store_grading: ${badVal}` });
    }

    // -----------------------------
    // read settings: quality_settings
    // -----------------------------
    const qsDoc = await settingCol.findOne({ type: 'quality_settings' });
    if (!qsDoc) return res.status(500).json({ ok: false, error: 'quality_settings not found' });

    const minTemp = Number(qsDoc?.ramen_standard?.min_temp_c);
    const maxProd = Number(qsDoc?.ramen_standard?.max_production_time_sec);
    if (!Number.isFinite(minTemp) || !Number.isFinite(maxProd)) {
      return res.status(500).json({ ok: false, error: 'quality_settings.ramen_standard invalid' });
    }

    const excellentMin = Number(qsDoc?.store_grading?.excellent_ratio_min);
    const watchMin = Number(qsDoc?.store_grading?.watch_ratio_min);
    if (!Number.isFinite(excellentMin) || !Number.isFinite(watchMin)) {
      return res.status(500).json({ ok: false, error: 'quality_settings.store_grading invalid' });
    }

    const mealPeriods = Array.isArray(qsDoc?.meal_periods) ? qsDoc.meal_periods.filter(p => p?.enabled !== false) : [];
    if (!mealPeriods.length) return res.status(500).json({ ok: false, error: 'quality_settings.meal_periods empty' });

    for (const p of mealPeriods) {
      if (!p?.key || !isHHMM(p.start) || !isHHMM(p.end)) {
        return res.status(500).json({ ok: false, error: `invalid meal_period item: ${JSON.stringify(p)}` });
      }
    }

    const mealPeriodByKey = Object.fromEntries(mealPeriods.map(p => [p.key, p]));
    const mpIndex = new Map(mealPeriods.map(p => [p.key, p]));

    // meal_period_key 驗證（若有指定）
    if (mealKeys) {
      for (const k of mealKeys) {
        const p = mpIndex.get(k);
        if (!p) return res.status(400).json({ ok: false, error: `invalid meal_period_key: ${k}` });
      }
    }

    // meal period switch branches for output meal_period_key
    const switchBranches = mealPeriods.map(p => {
      const [sh, sm] = p.start.split(':').map(n => parseInt(n, 10));
      const [eh, em] = p.end.split(':').map(n => parseInt(n, 10));
      const startMin = sh * 60 + sm;
      const endMin = eh * 60 + em;

      const cond = (endMin >= startMin)
        ? { $and: [{ $gte: ['$event_min', startMin] }, { $lt: ['$event_min', endMin] }] }
        : { $or:  [{ $gte: ['$event_min', startMin] }, { $lt: ['$event_min', endMin] }] };

      return { case: cond, then: p.key };
    });

    // =====================================================
    // sortStage
    // =====================================================
    let sortStage = null;
    switch (sortBy) {
      case 'production_time_desc':
        sortStage = { production_time_sec: -1, event_time: 1 };
        break;
      case 'serving_temp_asc':
        sortStage = { serving_temp: 1, event_time: 1 };
        break;
      case 'ingredient_unknown_first':
        // // 你原本這裡是 1（小到大），如果要「unknown 排前面」應該用 -1
        // sortStage = { ingredient_unknown: -1, event_time: 1 };
        // ✅ ramen6_first=1 排最前面，其它不限（再用 event_time 穩定一下）
        sortStage = { ramen6_first: -1, event_time: 1 };
        break;
      case 'event_time_desc':
        sortStage = { event_time: -1 };
        break;
      case 'event_time_asc':
      default:
        sortStage = { event_time: 1 };
        break;
    }

    // =====================================================
    // Pipeline
    // =====================================================
    const pipe = [];

    // base match
    const match = {};
    if (tFrom || tTo) {
      match.event_time = {};
      if (tFrom) match.event_time.$gte = tFrom;
      if (tTo)   match.event_time.$lte = tTo;
    }
    if (deviceIds) match.device_id = { $in: deviceIds };
    if (Object.keys(match).length) pipe.push({ $match: match });

    // keep original constraints
    pipe.push({
      $match: {
        production_time_sec: { $ne: null },
        serving_temp: { $ne: null }
      }
    });

    // compute event_min
    pipe.push(
      { $addFields: { hhmm: { $dateToString: { format: '%H:%M', date: '$event_time', timezone: tz } } } },
      {
        $addFields: {
          event_min: {
            $add: [
              { $multiply: [{ $toInt: { $substrBytes: ['$hhmm', 0, 2] } }, 60] },
              { $toInt: { $substrBytes: ['$hhmm', 3, 2] } }
            ]
          }
        }
      }
    );

    // meal_period_key filter (multi)
    if (mealKeys) {
      const ranges = mealKeys.map(k => {
        const p = mpIndex.get(k);
        const [sh, sm] = p.start.split(':').map(n => parseInt(n, 10));
        const [eh, em] = p.end.split(':').map(n => parseInt(n, 10));
        return { startMin: sh * 60 + sm, endMin: eh * 60 + em };
      });

      const orConds = [];
      for (const r of ranges) {
        if (r.endMin >= r.startMin) {
          orConds.push({ event_min: { $gte: r.startMin, $lt: r.endMin } });
        } else {
          orConds.push({ $or: [{ event_min: { $gte: r.startMin } }, { event_min: { $lt: r.endMin } }] });
        }
      }
      pipe.push({ $match: { $or: orConds } });
    }

    // join devices for store filters
    pipe.push(
      {
        $lookup: {
          from: 'devices',
          localField: 'device_id',
          foreignField: 'device_id',
          as: 'dv'
        }
      },
      { $unwind: { path: '$dv', preserveNullAndEmptyArrays: includeUnknown } }
    );

    // store filters
    const storeFilter = {};
    if (!includeUnknown) {
      storeFilter['dv.store.number'] = { $ne: null };
      storeFilter['dv.store.name'] = { $ne: null };
      storeFilter['dv.store.market_id'] = { $ne: null };
    }
    if (marketIds) storeFilter['dv.store.market_id'] = { $in: marketIds };
    if (storeNumbers) storeFilter['dv.store.number'] = { $in: storeNumbers };
    if (Object.keys(storeFilter).length) pipe.push({ $match: storeFilter });

    // compute ok flags + ingredient_unknown
    pipe.push(
      {
        $addFields: {
          temp_ok: { $gte: ['$serving_temp', minTemp] },
          prod_time_ok: { $lte: ['$production_time_sec', maxProd] },
          ingredient_ok: { $ne: ['$ingredient_missing', true] },

          ingredient_unknown: {
            $cond: [
              { $or: [{ $eq: ['$ingredient_missing', null] }, { $not: ['$ingredient_missing'] }] },
              1,
              0
            ]
          }
        }
      },
      { $addFields: { ok_all: { $and: ['$temp_ok', '$prod_time_ok', '$ingredient_ok'] } } }
    );

    // compute meal_period_key for output
    pipe.push({
      $addFields: {
        meal_period_key: {
          $switch: {
            branches: switchBranches,
            default: 'UNKNOWN_PERIOD'
          }
        }
      }
    });

    // ramen_key from raw.ramen_name
    pipe.push({
      $addFields: {
        ramen_key: {
          $ifNull: [
            '$raw.ramen_name',
            { $ifNull: ['$raw.ramenName', { $ifNull: ['$ramen_name', null] }] }
          ]
        }
      }
    });
    
    // ✅ 新增：只抓 ramen_name == 6 的排最前面
    pipe.push({
      $addFields: {
        ramen6_first: {
          $cond: [{ $eq: [{ $toString: '$ramen_key' }, '6'] }, 1, 0]
        }
      }
    });

    // =====================================================
    // store_grading facet path
    // =====================================================
    if (gradingFilter) {
      pipe.push({
        $facet: {
          store_ok: [
            {
              $group: {
                _id: '$dv.store.number',
                total: { $sum: 1 },
                ok: { $sum: { $cond: ['$ok_all', 1, 0] } }
              }
            },
            {
              $addFields: {
                ok_ratio: { $cond: [{ $gt: ['$total', 0] }, { $divide: ['$ok', '$total'] }, 0] }
              }
            },
            {
              $project: {
                _id: 0,
                store_number: '$_id',
                store_grading: {
                  $switch: {
                    branches: [
                      { case: { $gte: ['$ok_ratio', excellentMin] }, then: 'excellent' },
                      { case: { $gte: ['$ok_ratio', watchMin] }, then: 'watch' }
                    ],
                    default: 'bad'
                  }
                }
              }
            },
            { $match: { store_grading: { $in: gradingFilter } } }
          ],
          rows: [
            { $sort: sortStage },
            {
              $project: Object.assign(
                {
                  _id: 1,
                  time: { $dateToString: { format: '%Y-%m-%dT%H:%M:%SZ', date: '$event_time', timezone: 'UTC' } },

                  // ✅ 四捨五入：serving_temp 小數 1 位；production_time_sec 整數
                  serving_temp: { $round: ['$serving_temp', 1] },
                  production_time_sec: {
                    $let: {
                      vars: {
                        sec: { $toInt: { $round: ['$production_time_sec', 0] } }
                      },
                      in: {
                        $concat: [
                          {
                            $cond: [
                              { $lt: [{ $floor: { $divide: ['$$sec', 60] } }, 10] },
                              { $concat: ['0', { $toString: { $floor: { $divide: ['$$sec', 60] } } }] },
                              { $toString: { $floor: { $divide: ['$$sec', 60] } } }
                            ]
                          },
                          ':',
                          {
                            $cond: [
                              { $lt: [{ $mod: ['$$sec', 60] }, 10] },
                              { $concat: ['0', { $toString: { $mod: ['$$sec', 60] } }] },
                              { $toString: { $mod: ['$$sec', 60] } }
                            ]
                          }
                        ]
                      }
                    }
                  },

                  roi_id: 1,
                  ingredient_ok: 1,
                  ingredient_unknown: 1,
                  ramen_key: 1,
                  meal_period_key: 1,

                  store: {
                    name:   includeUnknown ? { $ifNull: ['$dv.store.name', 'UNKNOWN'] } : '$dv.store.name',
                    number: includeUnknown ? { $ifNull: ['$dv.store.number', 'UNKNOWN'] } : '$dv.store.number',
                    market: includeUnknown ? { $ifNull: ['$dv.store.market', 'UNKNOWN'] } : '$dv.store.market',
                    market_id: includeUnknown ? { $ifNull: ['$dv.store.market_id', 'UNKNOWN'] } : '$dv.store.market_id'
                  }
                },
                includeRaw ? { raw: '$raw' } : {}
              )
            }
          ]
        }
      });

      const pack = (await col.aggregate(pipe).toArray())[0] || { rows: [], store_ok: [] };
      const allowStores = new Set((pack.store_ok || []).map(x => x.store_number));

      let rows = (pack.rows || []).filter(r => allowStores.has(r?.store?.number));

      rows = rows.map(r => {
        const key = String(r.ramen_key || '').trim();
        const ramen_name = RAMEN_NAME_MAP[key] ?? RAMEN_NAME_MAP.__DEFAULT__;
        const p = mealPeriodByKey[r.meal_period_key];

        return {
          ...r,
          ramen_name,
          meal_period: p ? { key: p.key, name: p.name, start: p.start, end: p.end }
                         : { key: 'UNKNOWN_PERIOD', name: '未知餐期', start: null, end: null }
        };
      });

      return res.json({ ok: true, data: rows });
    }

    // =====================================================
    // no store_grading filter path
    // =====================================================
    pipe.push(
      { $sort: sortStage },
      {
        $project: Object.assign(
          {
            _id: 1,
            time: { $dateToString: { format: '%Y-%m-%dT%H:%M:%SZ', date: '$event_time', timezone: 'UTC' } },

            // ✅ 四捨五入：serving_temp 小數 1 位；production_time_sec 整數
            serving_temp: { $round: ['$serving_temp', 1] },
            production_time_sec: {
              $let: {
                vars: {
                  sec: { $toInt: { $round: ['$production_time_sec', 0] } }
                },
                in: {
                  $concat: [
                    {
                      $cond: [
                        { $lt: [{ $floor: { $divide: ['$$sec', 60] } }, 10] },
                        { $concat: ['0', { $toString: { $floor: { $divide: ['$$sec', 60] } } }] },
                        { $toString: { $floor: { $divide: ['$$sec', 60] } } }
                      ]
                    },
                    ':',
                    {
                      $cond: [
                        { $lt: [{ $mod: ['$$sec', 60] }, 10] },
                        { $concat: ['0', { $toString: { $mod: ['$$sec', 60] } }] },
                        { $toString: { $mod: ['$$sec', 60] } }
                      ]
                    }
                  ]
                }
              }
            },

            roi_id: 1,
            ingredient_ok: 1,
            ingredient_unknown: 1,
            ramen_key: 1,
            meal_period_key: 1,

            store: {
              name:   includeUnknown ? { $ifNull: ['$dv.store.name', 'UNKNOWN'] } : '$dv.store.name',
              number: includeUnknown ? { $ifNull: ['$dv.store.number', 'UNKNOWN'] } : '$dv.store.number',
              market: includeUnknown ? { $ifNull: ['$dv.store.market', 'UNKNOWN'] } : '$dv.store.market',
              market_id: includeUnknown ? { $ifNull: ['$dv.store.market_id', 'UNKNOWN'] } : '$dv.store.market_id'
            }
          },
          includeRaw ? { raw: '$raw' } : {}
        )
      }
    );

    let rows = await col.aggregate(pipe).toArray();

    rows = rows.map(r => {
      const key = String(r.ramen_key || '').trim();
      const ramen_name = RAMEN_NAME_MAP[key] ?? RAMEN_NAME_MAP.__DEFAULT__;
      const p = mealPeriodByKey[r.meal_period_key];

      return {
        ...r,
        ramen_name,
        meal_period: p ? { key: p.key, name: p.name, start: p.start, end: p.end }
                       : { key: 'UNKNOWN_PERIOD', name: '未知餐期', start: null, end: null }
      };
    });

    return res.json({ ok: true, data: rows });

  } catch (err) {
    console.error('[POST /quality/entries] error:', err);
    return res.status(500).json({ ok: false, error: String(err?.message || err) });
  }
});

// app.post('/entries', async (req, res) => {
//   try {
//     const body = req.body || {};

//     const includeRaw = ['1','true','True'].includes(String(body.include_raw || '0'));
//     const includeUnknown = ['1','true','True'].includes(String(body.include_unknown || '0'));

//     const tFrom = parseISO(body.from);
//     const tTo   = parseISO(body.to);
//     const tz    = String(body.tz || 'Asia/Shanghai');

//     // ✅ sort
//     const sortBy = String(body.sort_by || 'event_time_asc');

//     // filters (multi)
//     const marketIds    = parseMulti(body.market_id);
//     const storeNumbers = parseMulti(body.store_number, { allowAll: true });
//     const mealKeys     = parseMulti(body.meal_period_key, { allowAll: true });
//     const deviceIds    = parseMulti(body.device_id);

//     let gradingFilter = parseMulti(body.store_grading);
//     if (gradingFilter) {
//       const allowed = new Set(['excellent','watch','bad']);
//       const badVal = gradingFilter.find(x => !allowed.has(x));
//       if (badVal) return res.status(400).json({ ok: false, error: `invalid store_grading: ${badVal}` });
//     }

//     // -----------------------------
//     // read settings: quality_settings
//     // -----------------------------
//     const qsDoc = await settingCol.findOne({ type: 'quality_settings' });
//     if (!qsDoc) return res.status(500).json({ ok: false, error: 'quality_settings not found' });

//     const minTemp = Number(qsDoc?.ramen_standard?.min_temp_c);
//     const maxProd = Number(qsDoc?.ramen_standard?.max_production_time_sec);
//     if (!Number.isFinite(minTemp) || !Number.isFinite(maxProd)) {
//       return res.status(500).json({ ok: false, error: 'quality_settings.ramen_standard invalid' });
//     }

//     const excellentMin = Number(qsDoc?.store_grading?.excellent_ratio_min);
//     const watchMin = Number(qsDoc?.store_grading?.watch_ratio_min);
//     if (!Number.isFinite(excellentMin) || !Number.isFinite(watchMin)) {
//       return res.status(500).json({ ok: false, error: 'quality_settings.store_grading invalid' });
//     }

//     const mealPeriods = Array.isArray(qsDoc?.meal_periods) ? qsDoc.meal_periods.filter(p => p?.enabled !== false) : [];
//     if (!mealPeriods.length) return res.status(500).json({ ok: false, error: 'quality_settings.meal_periods empty' });

//     for (const p of mealPeriods) {
//       if (!p?.key || !isHHMM(p.start) || !isHHMM(p.end)) {
//         return res.status(500).json({ ok: false, error: `invalid meal_period item: ${JSON.stringify(p)}` });
//       }
//     }

//     const mealPeriodByKey = Object.fromEntries(mealPeriods.map(p => [p.key, p]));
//     const mpIndex = new Map(mealPeriods.map(p => [p.key, p]));

//     // meal_period_key 驗證（若有指定）
//     if (mealKeys) {
//       for (const k of mealKeys) {
//         const p = mpIndex.get(k);
//         if (!p) return res.status(400).json({ ok: false, error: `invalid meal_period_key: ${k}` });
//       }
//     }

//     // meal period switch branches for output meal_period_key
//     const switchBranches = mealPeriods.map(p => {
//       const [sh, sm] = p.start.split(':').map(n => parseInt(n, 10));
//       const [eh, em] = p.end.split(':').map(n => parseInt(n, 10));
//       const startMin = sh * 60 + sm;
//       const endMin = eh * 60 + em;

//       const cond = (endMin >= startMin)
//         ? { $and: [{ $gte: ['$event_min', startMin] }, { $lt: ['$event_min', endMin] }] }
//         : { $or:  [{ $gte: ['$event_min', startMin] }, { $lt: ['$event_min', endMin] }] };

//       return { case: cond, then: p.key };
//     });

//     // =====================================================
//     // sortStage（✅ 你要的三種排序）
//     // =====================================================
//     let sortStage = null;
//     switch (sortBy) {
//       case 'production_time_desc':
//         sortStage = { production_time_sec: -1, event_time: 1 };
//         break;
//       case 'serving_temp_asc':
//         sortStage = { serving_temp: 1, event_time: 1 };
//         break;
//       case 'ingredient_unknown_first':
//         sortStage = { ingredient_unknown: 1, event_time: 1 };
//         break;
//       case 'event_time_desc':
//         sortStage = { event_time: -1 };
//         break;
//       case 'event_time_asc':
//       default:
//         sortStage = { event_time: 1 };
//         break;
//     }

//     // =====================================================
//     // Pipeline
//     // =====================================================
//     const pipe = [];

//     // base match
//     const match = {};
//     if (tFrom || tTo) {
//       match.event_time = {};
//       if (tFrom) match.event_time.$gte = tFrom;
//       if (tTo)   match.event_time.$lte = tTo;
//     }
//     if (deviceIds) match.device_id = { $in: deviceIds };
//     if (Object.keys(match).length) pipe.push({ $match: match });

//     // keep original constraints
//     pipe.push({
//       $match: {
//         production_time_sec: { $ne: null },
//         serving_temp: { $ne: null }
//       }
//     });

//     // compute event_min
//     pipe.push(
//       { $addFields: { hhmm: { $dateToString: { format: '%H:%M', date: '$event_time', timezone: tz } } } },
//       {
//         $addFields: {
//           event_min: {
//             $add: [
//               { $multiply: [{ $toInt: { $substrBytes: ['$hhmm', 0, 2] } }, 60] },
//               { $toInt: { $substrBytes: ['$hhmm', 3, 2] } }
//             ]
//           }
//         }
//       }
//     );

//     // meal_period_key filter (multi)
//     if (mealKeys) {
//       const ranges = mealKeys.map(k => {
//         const p = mpIndex.get(k);
//         const [sh, sm] = p.start.split(':').map(n => parseInt(n, 10));
//         const [eh, em] = p.end.split(':').map(n => parseInt(n, 10));
//         return { startMin: sh * 60 + sm, endMin: eh * 60 + em };
//       });

//       const orConds = [];
//       for (const r of ranges) {
//         if (r.endMin >= r.startMin) {
//           orConds.push({ event_min: { $gte: r.startMin, $lt: r.endMin } });
//         } else {
//           orConds.push({ $or: [{ event_min: { $gte: r.startMin } }, { event_min: { $lt: r.endMin } }] });
//         }
//       }
//       pipe.push({ $match: { $or: orConds } });
//     }

//     // join devices for store filters
//     pipe.push(
//       {
//         $lookup: {
//           from: 'devices',
//           localField: 'device_id',
//           foreignField: 'device_id',
//           as: 'dv'
//         }
//       },
//       { $unwind: { path: '$dv', preserveNullAndEmptyArrays: includeUnknown } }
//     );

//     // store filters
//     const storeFilter = {};
//     if (!includeUnknown) {
//       storeFilter['dv.store.number'] = { $ne: null };
//       storeFilter['dv.store.name'] = { $ne: null };
//       storeFilter['dv.store.market_id'] = { $ne: null };
//     }
//     if (marketIds) storeFilter['dv.store.market_id'] = { $in: marketIds };
//     if (storeNumbers) storeFilter['dv.store.number'] = { $in: storeNumbers };
//     if (Object.keys(storeFilter).length) pipe.push({ $match: storeFilter });

//     // compute ok flags + ✅ ingredient_unknown（無法辨識排前）
//     pipe.push(
//       {
//         $addFields: {
//           temp_ok: { $gte: ['$serving_temp', minTemp] },
//           prod_time_ok: { $lte: ['$production_time_sec', maxProd] },
//           ingredient_ok: { $ne: ['$ingredient_missing', true] },

//           // ✅ 無法辨識：ingredient_missing 欄位不存在 / 為 null
//           ingredient_unknown: {
//             $cond: [
//               {
//                 $or: [
//                   { $eq: ['$ingredient_missing', null] },
//                   { $not: ['$ingredient_missing'] }
//                 ]
//               },
//               1,
//               0
//             ]
//           }
//         }
//       },
//       { $addFields: { ok_all: { $and: ['$temp_ok', '$prod_time_ok', '$ingredient_ok'] } } }
//     );

//     // compute meal_period_key for output
//     pipe.push({
//       $addFields: {
//         meal_period_key: {
//           $switch: {
//             branches: switchBranches,
//             default: 'UNKNOWN_PERIOD'
//           }
//         }
//       }
//     });

//     // ====== ✅ 這裡新增：從 raw.ramen_name 抽出來（不管 includeRaw）
//     // 你後面要用它 map 實際拉麵名稱
//     pipe.push({
//       $addFields: {
//         ramen_key: {
//           $ifNull: [
//             '$raw.ramen_name',
//             { $ifNull: ['$raw.ramenName', { $ifNull: ['$ramen_name', null] }] }
//           ]
//         }
//       }
//     });

//     // =====================================================
//     // store_grading facet path
//     // =====================================================
//     if (gradingFilter) {
//       pipe.push({
//         $facet: {
//           store_ok: [
//             {
//               $group: {
//                 _id: '$dv.store.number',
//                 total: { $sum: 1 },
//                 ok: { $sum: { $cond: ['$ok_all', 1, 0] } }
//               }
//             },
//             {
//               $addFields: {
//                 ok_ratio: { $cond: [{ $gt: ['$total', 0] }, { $divide: ['$ok', '$total'] }, 0] }
//               }
//             },
//             {
//               $project: {
//                 _id: 0,
//                 store_number: '$_id',
//                 store_grading: {
//                   $switch: {
//                     branches: [
//                       { case: { $gte: ['$ok_ratio', excellentMin] }, then: 'excellent' },
//                       { case: { $gte: ['$ok_ratio', watchMin] }, then: 'watch' }
//                     ],
//                     default: 'bad'
//                   }
//                 }
//               }
//             },
//             { $match: { store_grading: { $in: gradingFilter } } }
//           ],
//           rows: [
//             { $sort: sortStage },
//             {
//               $project: Object.assign(
//                 {
//                   _id: 1,
//                   time: { $dateToString: { format: '%Y-%m-%dT%H:%M:%SZ', date: '$event_time', timezone: 'UTC' } },
//                   serving_temp: 1,
//                   production_time_sec: 1,
//                   roi_id: 1,
//                   ingredient_ok: 1,
//                   ingredient_unknown: 1,

//                   // ✅ 給後面 map 用（不管 includeRaw）
//                   ramen_key: 1,

//                   meal_period_key: 1,

//                   store: {
//                     name:   includeUnknown ? { $ifNull: ['$dv.store.name', 'UNKNOWN'] } : '$dv.store.name',
//                     number: includeUnknown ? { $ifNull: ['$dv.store.number', 'UNKNOWN'] } : '$dv.store.number',
//                     market: includeUnknown ? { $ifNull: ['$dv.store.market', 'UNKNOWN'] } : '$dv.store.market',
//                     market_id: includeUnknown ? { $ifNull: ['$dv.store.market_id', 'UNKNOWN'] } : '$dv.store.market_id'
//                   }
//                 },
//                 includeRaw ? { raw: '$raw' } : {}
//               )
//             }
//           ]
//         }
//       });

//       const pack = (await col.aggregate(pipe).toArray())[0] || { rows: [], store_ok: [] };
//       const allowStores = new Set((pack.store_ok || []).map(x => x.store_number));

//       let rows = (pack.rows || []).filter(r => allowStores.has(r?.store?.number));

//       rows = rows.map(r => {
//         const key = String(r.ramen_key || '').trim();
//         const ramen_name = RAMEN_NAME_MAP[key] ?? RAMEN_NAME_MAP.__DEFAULT__;

//         const p = mealPeriodByKey[r.meal_period_key];
//         return {
//           ...r,
//           ramen_name,
//           meal_period: p ? { key: p.key, name: p.name, start: p.start, end: p.end }
//                          : { key: 'UNKNOWN_PERIOD', name: '未知餐期', start: null, end: null }
//         };
//       });

//       return res.json({ ok: true, data: rows });
//     }

//     // =====================================================
//     // no store_grading filter path
//     // =====================================================
//     pipe.push(
//       { $sort: sortStage },
//       {
//         $project: Object.assign(
//           {
//             _id: 1,
//             time: { $dateToString: { format: '%Y-%m-%dT%H:%M:%SZ', date: '$event_time', timezone: 'UTC' } },
//             // serving_temp: 1,
//             // production_time_sec: 1,
//             // ✅ 四捨五入：serving_temp 小數 1 位；production_time_sec 整數
//             serving_temp: { $round: ['$serving_temp', 1] },
//             production_time_sec: { $round: ['$production_time_sec', 1] },
//             roi_id: 1,
//             ingredient_ok: 1,
//             ingredient_unknown: 1,

//             // ✅ 給後面 map 用（不管 includeRaw）
//             ramen_key: 1,

//             meal_period_key: 1,

//             store: {
//               name:   includeUnknown ? { $ifNull: ['$dv.store.name', 'UNKNOWN'] } : '$dv.store.name',
//               number: includeUnknown ? { $ifNull: ['$dv.store.number', 'UNKNOWN'] } : '$dv.store.number',
//               market: includeUnknown ? { $ifNull: ['$dv.store.market', 'UNKNOWN'] } : '$dv.store.market',
//               market_id: includeUnknown ? { $ifNull: ['$dv.store.market_id', 'UNKNOWN'] } : '$dv.store.market_id'
//             }
//           },
//           includeRaw ? { raw: '$raw' } : {}
//         )
//       }
//     );

//     let rows = await col.aggregate(pipe).toArray();

//     rows = rows.map(r => {
//       const key = String(r.ramen_key || '').trim();
//       const ramen_name = RAMEN_NAME_MAP[key] ?? RAMEN_NAME_MAP.__DEFAULT__;
//       const p = mealPeriodByKey[r.meal_period_key];

//       return {
//         ...r,
//         ramen_name,
//         meal_period: p ? { key: p.key, name: p.name, start: p.start, end: p.end }
//                        : { key: 'UNKNOWN_PERIOD', name: '未知餐期', start: null, end: null }
//       };
//     });

//     return res.json({ ok: true, data: rows });

//   } catch (err) {
//     console.error('[POST /quality/entries] error:', err);
//     return res.status(500).json({ ok: false, error: String(err?.message || err) });
//   }
// });

// app.post('/entries', async (req, res) => {
//   try {
//     const body = req.body || {};

//     const includeRaw = ['1','true','True'].includes(String(body.include_raw || '0'));
//     const includeUnknown = ['1','true','True'].includes(String(body.include_unknown || '0'));

//     const tFrom = parseISO(body.from);
//     const tTo   = parseISO(body.to);
//     const tz    = String(body.tz || 'Asia/Shanghai');

//     // ✅ sort
//     const sortBy = String(body.sort_by || 'event_time_asc');

//     // filters (multi)
//     const marketIds    = parseMulti(body.market_id);
//     const storeNumbers = parseMulti(body.store_number, { allowAll: true });
//     const mealKeys     = parseMulti(body.meal_period_key, { allowAll: true });
//     const deviceIds    = parseMulti(body.device_id);

//     let gradingFilter = parseMulti(body.store_grading);
//     if (gradingFilter) {
//       const allowed = new Set(['excellent','watch','bad']);
//       const badVal = gradingFilter.find(x => !allowed.has(x));
//       if (badVal) return res.status(400).json({ ok: false, error: `invalid store_grading: ${badVal}` });
//     }

//     // -----------------------------
//     // read settings: quality_settings
//     // -----------------------------
//     const qsDoc = await settingCol.findOne({ type: 'quality_settings' });
//     if (!qsDoc) return res.status(500).json({ ok: false, error: 'quality_settings not found' });

//     const minTemp = Number(qsDoc?.ramen_standard?.min_temp_c);
//     const maxProd = Number(qsDoc?.ramen_standard?.max_production_time_sec);
//     if (!Number.isFinite(minTemp) || !Number.isFinite(maxProd)) {
//       return res.status(500).json({ ok: false, error: 'quality_settings.ramen_standard invalid' });
//     }

//     const excellentMin = Number(qsDoc?.store_grading?.excellent_ratio_min);
//     const watchMin = Number(qsDoc?.store_grading?.watch_ratio_min);
//     if (!Number.isFinite(excellentMin) || !Number.isFinite(watchMin)) {
//       return res.status(500).json({ ok: false, error: 'quality_settings.store_grading invalid' });
//     }

//     const mealPeriods = Array.isArray(qsDoc?.meal_periods) ? qsDoc.meal_periods.filter(p => p?.enabled !== false) : [];
//     if (!mealPeriods.length) return res.status(500).json({ ok: false, error: 'quality_settings.meal_periods empty' });

//     for (const p of mealPeriods) {
//       if (!p?.key || !isHHMM(p.start) || !isHHMM(p.end)) {
//         return res.status(500).json({ ok: false, error: `invalid meal_period item: ${JSON.stringify(p)}` });
//       }
//     }

//     const mealPeriodByKey = Object.fromEntries(mealPeriods.map(p => [p.key, p]));
//     const mpIndex = new Map(mealPeriods.map(p => [p.key, p]));

//     // meal_period_key 驗證（若有指定）
//     if (mealKeys) {
//       for (const k of mealKeys) {
//         const p = mpIndex.get(k);
//         if (!p) return res.status(400).json({ ok: false, error: `invalid meal_period_key: ${k}` });
//       }
//     }

//     // meal period switch branches for output meal_period_key
//     const switchBranches = mealPeriods.map(p => {
//       const [sh, sm] = p.start.split(':').map(n => parseInt(n, 10));
//       const [eh, em] = p.end.split(':').map(n => parseInt(n, 10));
//       const startMin = sh * 60 + sm;
//       const endMin = eh * 60 + em;

//       const cond = (endMin >= startMin)
//         ? { $and: [{ $gte: ['$event_min', startMin] }, { $lt: ['$event_min', endMin] }] }
//         : { $or:  [{ $gte: ['$event_min', startMin] }, { $lt: ['$event_min', endMin] }] };

//       return { case: cond, then: p.key };
//     });

//     // =====================================================
//     // sortStage（✅ 你要的三種排序）
//     // =====================================================
//     let sortStage = null;
//     switch (sortBy) {
//       case 'production_time_desc':
//         sortStage = { production_time_sec: -1, event_time: 1 };
//         break;
//       case 'serving_temp_asc':
//         sortStage = { serving_temp: 1, event_time: 1 };
//         break;
//       case 'ingredient_unknown_first':
//         sortStage = { ingredient_unknown: -1, event_time: 1 };
//         break;
//       case 'event_time_desc':
//         sortStage = { event_time: -1 };
//         break;
//       case 'event_time_asc':
//       default:
//         sortStage = { event_time: 1 };
//         break;
//     }

//     // =====================================================
//     // Pipeline
//     // =====================================================
//     const pipe = [];

//     // base match
//     const match = {};
//     if (tFrom || tTo) {
//       match.event_time = {};
//       if (tFrom) match.event_time.$gte = tFrom;
//       if (tTo)   match.event_time.$lte = tTo;
//     }
//     if (deviceIds) match.device_id = { $in: deviceIds };
//     if (Object.keys(match).length) pipe.push({ $match: match });

//     // keep original constraints
//     pipe.push({
//       $match: {
//         production_time_sec: { $ne: null },
//         serving_temp: { $ne: null }
//       }
//     });

//     // compute event_min
//     pipe.push(
//       { $addFields: { hhmm: { $dateToString: { format: '%H:%M', date: '$event_time', timezone: tz } } } },
//       {
//         $addFields: {
//           event_min: {
//             $add: [
//               { $multiply: [{ $toInt: { $substrBytes: ['$hhmm', 0, 2] } }, 60] },
//               { $toInt: { $substrBytes: ['$hhmm', 3, 2] } }
//             ]
//           }
//         }
//       }
//     );

//     // meal_period_key filter (multi)
//     if (mealKeys) {
//       const ranges = mealKeys.map(k => {
//         const p = mpIndex.get(k);
//         const [sh, sm] = p.start.split(':').map(n => parseInt(n, 10));
//         const [eh, em] = p.end.split(':').map(n => parseInt(n, 10));
//         return { startMin: sh * 60 + sm, endMin: eh * 60 + em };
//       });

//       const orConds = [];
//       for (const r of ranges) {
//         if (r.endMin >= r.startMin) {
//           orConds.push({ event_min: { $gte: r.startMin, $lt: r.endMin } });
//         } else {
//           orConds.push({ $or: [{ event_min: { $gte: r.startMin } }, { event_min: { $lt: r.endMin } }] });
//         }
//       }
//       pipe.push({ $match: { $or: orConds } });
//     }

//     // join devices for store filters
//     pipe.push(
//       {
//         $lookup: {
//           from: 'devices',
//           localField: 'device_id',
//           foreignField: 'device_id',
//           as: 'dv'
//         }
//       },
//       { $unwind: { path: '$dv', preserveNullAndEmptyArrays: includeUnknown } }
//     );

//     // store filters
//     const storeFilter = {};
//     if (!includeUnknown) {
//       storeFilter['dv.store.number'] = { $ne: null };
//       storeFilter['dv.store.name'] = { $ne: null };
//       storeFilter['dv.store.market_id'] = { $ne: null };
//     }
//     if (marketIds) storeFilter['dv.store.market_id'] = { $in: marketIds };
//     if (storeNumbers) storeFilter['dv.store.number'] = { $in: storeNumbers };
//     if (Object.keys(storeFilter).length) pipe.push({ $match: storeFilter });

//     // compute ok flags + ✅ ingredient_unknown（無法辨識排前）
//     pipe.push(
//       {
//         $addFields: {
//           temp_ok: { $gte: ['$serving_temp', minTemp] },
//           prod_time_ok: { $lte: ['$production_time_sec', maxProd] },
//           ingredient_ok: { $ne: ['$ingredient_missing', true] },

//           // ✅ 無法辨識：ingredient_missing 欄位不存在 / 為 null
//           ingredient_unknown: {
//             $cond: [
//               {
//                 $or: [
//                   { $eq: ['$ingredient_missing', null] },
//                   { $not: ['$ingredient_missing'] }
//                 ]
//               },
//               1,
//               0
//             ]
//           }
//         }
//       },
//       { $addFields: { ok_all: { $and: ['$temp_ok', '$prod_time_ok', '$ingredient_ok'] } } }
//     );

//     // compute meal_period_key for output
//     pipe.push({
//       $addFields: {
//         meal_period_key: {
//           $switch: {
//             branches: switchBranches,
//             default: 'UNKNOWN_PERIOD'
//           }
//         }
//       }
//     });

//     // =====================================================
//     // store_grading facet path
//     // =====================================================
//     if (gradingFilter) {
//       pipe.push({
//         $facet: {
//           store_ok: [
//             {
//               $group: {
//                 _id: '$dv.store.number',
//                 total: { $sum: 1 },
//                 ok: { $sum: { $cond: ['$ok_all', 1, 0] } }
//               }
//             },
//             {
//               $addFields: {
//                 ok_ratio: { $cond: [{ $gt: ['$total', 0] }, { $divide: ['$ok', '$total'] }, 0] }
//               }
//             },
//             {
//               $project: {
//                 _id: 0,
//                 store_number: '$_id',
//                 store_grading: {
//                   $switch: {
//                     branches: [
//                       { case: { $gte: ['$ok_ratio', excellentMin] }, then: 'excellent' },
//                       { case: { $gte: ['$ok_ratio', watchMin] }, then: 'watch' }
//                     ],
//                     default: 'bad'
//                   }
//                 }
//               }
//             },
//             { $match: { store_grading: { $in: gradingFilter } } }
//           ],
//           rows: [
//             { $sort: sortStage }, // ✅ 動態排序
//             {
//               $project: Object.assign(
//                 {
//                   _id: 1,
//                   time: { $dateToString: { format: '%Y-%m-%dT%H:%M:%SZ', date: '$event_time', timezone: 'UTC' } },
//                   serving_temp: 1,
//                   production_time_sec: 1,
//                   roi_id: 1,
//                   ingredient_ok: 1,
//                   ingredient_unknown: 1,

//                   ramen_id: {
//                     $ifNull: [
//                       '$ramen_id',
//                       { $ifNull: ['$raw.ramen_id', { $ifNull: ['$raw.class_id', { $ifNull: ['$raw.classId', '$raw.class'] }] }] }
//                     ]
//                   },

//                   meal_period_key: 1,

//                   store: {
//                     name:   includeUnknown ? { $ifNull: ['$dv.store.name', 'UNKNOWN'] } : '$dv.store.name',
//                     number: includeUnknown ? { $ifNull: ['$dv.store.number', 'UNKNOWN'] } : '$dv.store.number',
//                     market: includeUnknown ? { $ifNull: ['$dv.store.market', 'UNKNOWN'] } : '$dv.store.market',
//                     market_id: includeUnknown ? { $ifNull: ['$dv.store.market_id', 'UNKNOWN'] } : '$dv.store.market_id'
//                   }
//                 },
//                 includeRaw ? { raw: '$raw' } : {}
//               )
//             }
//           ]
//         }
//       });

//       const pack = (await col.aggregate(pipe).toArray())[0] || { rows: [], store_ok: [] };
//       const allowStores = new Set((pack.store_ok || []).map(x => x.store_number));

//       let rows = (pack.rows || []).filter(r => allowStores.has(r?.store?.number));

//       rows = rows.map(r => {
//         const idNum = Number(r.ramen_id);
//         const ramen_name = RAMEN_NAME_MAP[idNum] ?? RAMEN_NAME_MAP[6];
//         const p = mealPeriodByKey[r.meal_period_key];

//         return {
//           ...r,
//           ramen_name,
//           meal_period: p ? { key: p.key, name: p.name, start: p.start, end: p.end }
//                          : { key: 'UNKNOWN_PERIOD', name: '未知餐期', start: null, end: null }
//         };
//       });

//       return res.json({ ok: true, data: rows });
//     }

//     // =====================================================
//     // no store_grading filter path
//     // =====================================================
//     pipe.push(
//       { $sort: sortStage }, // ✅ 動態排序
//       {
//         $project: Object.assign(
//           {
//             _id: 1,
//             time: { $dateToString: { format: '%Y-%m-%dT%H:%M:%SZ', date: '$event_time', timezone: 'UTC' } },
//             serving_temp: 1,
//             production_time_sec: 1,
//             roi_id: 1,
//             ingredient_ok: 1,
//             ingredient_unknown: 1,

//             ramen_id: {
//               $ifNull: [
//                 '$ramen_id',
//                 { $ifNull: ['$raw.ramen_id', { $ifNull: ['$raw.class_id', { $ifNull: ['$raw.classId', '$raw.class'] }] }] }
//               ]
//             },

//             meal_period_key: 1,

//             store: {
//               name:   includeUnknown ? { $ifNull: ['$dv.store.name', 'UNKNOWN'] } : '$dv.store.name',
//               number: includeUnknown ? { $ifNull: ['$dv.store.number', 'UNKNOWN'] } : '$dv.store.number',
//               market: includeUnknown ? { $ifNull: ['$dv.store.market', 'UNKNOWN'] } : '$dv.store.market',
//               market_id: includeUnknown ? { $ifNull: ['$dv.store.market_id', 'UNKNOWN'] } : '$dv.store.market_id'
//             }
//           },
//           includeRaw ? { raw: '$raw' } : {}
//         )
//       }
//     );

//     let rows = await col.aggregate(pipe).toArray();

//     rows = rows.map(r => {
//       const idNum = Number(r.ramen_id);
//       const ramen_name = RAMEN_NAME_MAP[idNum] ?? RAMEN_NAME_MAP[6];
//       const p = mealPeriodByKey[r.meal_period_key];

//       return {
//         ...r,
//         ramen_name,
//         meal_period: p ? { key: p.key, name: p.name, start: p.start, end: p.end }
//                        : { key: 'UNKNOWN_PERIOD', name: '未知餐期', start: null, end: null }
//       };
//     });

//     return res.json({ ok: true, data: rows });

//   } catch (err) {
//     console.error('[POST /quality/entries] error:', err);
//     return res.status(500).json({ ok: false, error: String(err?.message || err) });
//   }
// });

// app.post('/entries', async (req, res) => {
//   try {
//     const body = req.body || {};

//     const includeRaw = ['1','true','True'].includes(String(body.include_raw || '0'));
//     const includeUnknown = ['1','true','True'].includes(String(body.include_unknown || '0'));

//     const tFrom = parseISO(body.from);
//     const tTo   = parseISO(body.to);
//     const tz    = String(body.tz || 'Asia/Shanghai');

//     // filters (multi)
//     const marketIds    = parseMulti(body.market_id);
//     const storeNumbers = parseMulti(body.store_number, { allowAll: true });
//     const mealKeys     = parseMulti(body.meal_period_key, { allowAll: true });
//     const deviceIds    = parseMulti(body.device_id);

//     let gradingFilter = parseMulti(body.store_grading);
//     if (gradingFilter) {
//       const allowed = new Set(['excellent','watch','bad']);
//       const badVal = gradingFilter.find(x => !allowed.has(x));
//       if (badVal) return res.status(400).json({ ok: false, error: `invalid store_grading: ${badVal}` });
//     }

//     // -----------------------------
//     // read settings: quality_settings
//     // -----------------------------
//     const qsDoc = await settingCol.findOne({ type: 'quality_settings' });
//     if (!qsDoc) return res.status(500).json({ ok: false, error: 'quality_settings not found' });

//     const minTemp = Number(qsDoc?.ramen_standard?.min_temp_c);
//     const maxProd = Number(qsDoc?.ramen_standard?.max_production_time_sec);
//     if (!Number.isFinite(minTemp) || !Number.isFinite(maxProd)) {
//       return res.status(500).json({ ok: false, error: 'quality_settings.ramen_standard invalid' });
//     }

//     const excellentMin = Number(qsDoc?.store_grading?.excellent_ratio_min);
//     const watchMin = Number(qsDoc?.store_grading?.watch_ratio_min);
//     if (!Number.isFinite(excellentMin) || !Number.isFinite(watchMin)) {
//       return res.status(500).json({ ok: false, error: 'quality_settings.store_grading invalid' });
//     }

//     const mealPeriods = Array.isArray(qsDoc?.meal_periods) ? qsDoc.meal_periods.filter(p => p?.enabled !== false) : [];
//     if (!mealPeriods.length) return res.status(500).json({ ok: false, error: 'quality_settings.meal_periods empty' });

//     for (const p of mealPeriods) {
//       if (!p?.key || !isHHMM(p.start) || !isHHMM(p.end)) {
//         return res.status(500).json({ ok: false, error: `invalid meal_period item: ${JSON.stringify(p)}` });
//       }
//     }

//     const mealPeriodByKey = Object.fromEntries(mealPeriods.map(p => [p.key, p]));
//     const mpIndex = new Map(mealPeriods.map(p => [p.key, p]));

//     // meal_period_key 驗證（若有指定）
//     if (mealKeys) {
//       for (const k of mealKeys) {
//         const p = mpIndex.get(k);
//         if (!p) return res.status(400).json({ ok: false, error: `invalid meal_period_key: ${k}` });
//       }
//     }

//     // meal period switch branches for output meal_period_key
//     const switchBranches = mealPeriods.map(p => {
//       const [sh, sm] = p.start.split(':').map(n => parseInt(n, 10));
//       const [eh, em] = p.end.split(':').map(n => parseInt(n, 10));
//       const startMin = sh * 60 + sm;
//       const endMin = eh * 60 + em;

//       const cond = (endMin >= startMin)
//         ? { $and: [{ $gte: ['$event_min', startMin] }, { $lt: ['$event_min', endMin] }] }
//         : { $or:  [{ $gte: ['$event_min', startMin] }, { $lt: ['$event_min', endMin] }] };

//       return { case: cond, then: p.key };
//     });

//     // =====================================================
//     // Pipeline
//     // =====================================================
//     const pipe = [];

//     // base match
//     const match = {};
//     if (tFrom || tTo) {
//       match.event_time = {};
//       if (tFrom) match.event_time.$gte = tFrom;
//       if (tTo)   match.event_time.$lte = tTo;
//     }
//     if (deviceIds) match.device_id = { $in: deviceIds };
//     if (Object.keys(match).length) pipe.push({ $match: match });

//     // keep original constraints (you had these)
//     pipe.push({
//       $match: {
//         production_time_sec: { $ne: null },
//         serving_temp: { $ne: null }
//       }
//     });

//     // compute event_min
//     pipe.push(
//       { $addFields: { hhmm: { $dateToString: { format: '%H:%M', date: '$event_time', timezone: tz } } } },
//       {
//         $addFields: {
//           event_min: {
//             $add: [
//               { $multiply: [{ $toInt: { $substrBytes: ['$hhmm', 0, 2] } }, 60] },
//               { $toInt: { $substrBytes: ['$hhmm', 3, 2] } }
//             ]
//           }
//         }
//       }
//     );

//     // meal_period_key filter (multi) using OR ranges
//     if (mealKeys) {
//       const ranges = mealKeys.map(k => {
//         const p = mpIndex.get(k);
//         const [sh, sm] = p.start.split(':').map(n => parseInt(n, 10));
//         const [eh, em] = p.end.split(':').map(n => parseInt(n, 10));
//         return { startMin: sh * 60 + sm, endMin: eh * 60 + em };
//       });

//       const orConds = [];
//       for (const r of ranges) {
//         if (r.endMin >= r.startMin) {
//           orConds.push({ event_min: { $gte: r.startMin, $lt: r.endMin } });
//         } else {
//           orConds.push({ $or: [{ event_min: { $gte: r.startMin } }, { event_min: { $lt: r.endMin } }] });
//         }
//       }
//       pipe.push({ $match: { $or: orConds } });
//     }

//     // join devices for store filters
//     pipe.push(
//       {
//         $lookup: {
//           from: 'devices',
//           localField: 'device_id',
//           foreignField: 'device_id',
//           as: 'dv'
//         }
//       },
//       { $unwind: { path: '$dv', preserveNullAndEmptyArrays: includeUnknown } }
//     );

//     // store filters
//     const storeFilter = {};
//     if (!includeUnknown) {
//       storeFilter['dv.store.number'] = { $ne: null };
//       storeFilter['dv.store.name'] = { $ne: null };
//       storeFilter['dv.store.market_id'] = { $ne: null };
//     }
//     if (marketIds) storeFilter['dv.store.market_id'] = { $in: marketIds };
//     if (storeNumbers) storeFilter['dv.store.number'] = { $in: storeNumbers };
//     if (Object.keys(storeFilter).length) pipe.push({ $match: storeFilter });

//     // compute ok flags (for store grading calc)
//     pipe.push(
//       {
//         $addFields: {
//           temp_ok: { $gte: ['$serving_temp', minTemp] },
//           prod_time_ok: { $lte: ['$production_time_sec', maxProd] },
//           ingredient_ok: { $ne: ['$ingredient_missing', true] }
//         }
//       },
//       { $addFields: { ok_all: { $and: ['$temp_ok', '$prod_time_ok', '$ingredient_ok'] } } }
//     );

//     // compute meal_period_key for output
//     pipe.push({
//       $addFields: {
//         meal_period_key: {
//           $switch: {
//             branches: switchBranches,
//             default: 'UNKNOWN_PERIOD'
//           }
//         }
//       }
//     });

//     // =====================================================
//     // If store_grading filter is specified:
//     //  - facet store_ok: compute ok_ratio per store -> grading -> pick stores
//     //  - facet rows: output entries
//     // Then filter rows in Node by picked stores.
//     // =====================================================
//     if (gradingFilter) {
//       pipe.push({
//         $facet: {
//           store_ok: [
//             {
//               $group: {
//                 _id: '$dv.store.number',
//                 total: { $sum: 1 },
//                 ok: { $sum: { $cond: ['$ok_all', 1, 0] } }
//               }
//             },
//             {
//               $addFields: {
//                 ok_ratio: { $cond: [{ $gt: ['$total', 0] }, { $divide: ['$ok', '$total'] }, 0] }
//               }
//             },
//             {
//               $project: {
//                 _id: 0,
//                 store_number: '$_id',
//                 store_grading: {
//                   $switch: {
//                     branches: [
//                       { case: { $gte: ['$ok_ratio', excellentMin] }, then: 'excellent' },
//                       { case: { $gte: ['$ok_ratio', watchMin] }, then: 'watch' }
//                     ],
//                     default: 'bad'
//                   }
//                 }
//               }
//             },
//             { $match: { store_grading: { $in: gradingFilter } } }
//           ],
//           rows: [
//             { $sort: { event_time: 1 } },
//             {
//               $project: Object.assign(
//                 {
//                   _id: 1,
//                   time: { $dateToString: { format: '%Y-%m-%dT%H:%M:%SZ', date: '$event_time', timezone: 'UTC' } },
//                   serving_temp: 1,
//                   production_time_sec: 1,
//                   roi_id: 1,

//                   // ramen id source (robust)
//                   ramen_id: {
//                     $ifNull: [
//                       '$ramen_id',
//                       { $ifNull: ['$raw.ramen_id', { $ifNull: ['$raw.class_id', { $ifNull: ['$raw.classId', '$raw.class'] }] }] }
//                     ]
//                   },

//                   meal_period_key: 1,

//                   store: {
//                     name:   includeUnknown ? { $ifNull: ['$dv.store.name', 'UNKNOWN'] } : '$dv.store.name',
//                     number: includeUnknown ? { $ifNull: ['$dv.store.number', 'UNKNOWN'] } : '$dv.store.number',
//                     market: includeUnknown ? { $ifNull: ['$dv.store.market', 'UNKNOWN'] } : '$dv.store.market',
//                     market_id: includeUnknown ? { $ifNull: ['$dv.store.market_id', 'UNKNOWN'] } : '$dv.store.market_id'
//                   }
//                 },
//                 includeRaw ? { raw: '$raw' } : {}
//               )
//             }
//           ]
//         }
//       });

//       const pack = (await col.aggregate(pipe).toArray())[0] || { rows: [], store_ok: [] };
//       const allowStores = new Set((pack.store_ok || []).map(x => x.store_number));

//       let rows = (pack.rows || []).filter(r => allowStores.has(r?.store?.number));

//       // Node-side enrich: ramen_name + meal_period
//       rows = rows.map(r => {
//         const idNum = Number(r.ramen_id);
//         const ramen_name = RAMEN_NAME_MAP[idNum] ?? RAMEN_NAME_MAP[6];
//         const p = mealPeriodByKey[r.meal_period_key];

//         return {
//           ...r,
//           ramen_name,
//           meal_period: p ? { key: p.key, name: p.name, start: p.start, end: p.end } 
//                          : { key: 'UNKNOWN_PERIOD', name: '未知餐期', start: null, end: null }
//         };
//       });

//       return res.json({ ok: true, data: rows });
//     }

//     // =====================================================
//     // No store_grading filter: output rows directly
//     // =====================================================
//     pipe.push(
//       { $sort: { event_time: 1 } },
//       {
//         $project: Object.assign(
//           {
//             _id: 1,
//             time: { $dateToString: { format: '%Y-%m-%dT%H:%M:%SZ', date: '$event_time', timezone: 'UTC' } },
//             serving_temp: 1,
//             production_time_sec: 1,
//             roi_id: 1,

//             ramen_id: {
//               $ifNull: [
//                 '$ramen_id',
//                 { $ifNull: ['$raw.ramen_id', { $ifNull: ['$raw.class_id', { $ifNull: ['$raw.classId', '$raw.class'] }] }] }
//               ]
//             },

//             meal_period_key: 1,

//             store: {
//               name:   includeUnknown ? { $ifNull: ['$dv.store.name', 'UNKNOWN'] } : '$dv.store.name',
//               number: includeUnknown ? { $ifNull: ['$dv.store.number', 'UNKNOWN'] } : '$dv.store.number',
//               market: includeUnknown ? { $ifNull: ['$dv.store.market', 'UNKNOWN'] } : '$dv.store.market',
//               market_id: includeUnknown ? { $ifNull: ['$dv.store.market_id', 'UNKNOWN'] } : '$dv.store.market_id'
//             }
//           },
//           includeRaw ? { raw: '$raw' } : {}
//         )
//       }
//     );

//     let rows = await col.aggregate(pipe).toArray();

//     rows = rows.map(r => {
//       const idNum = Number(r.ramen_id);
//       const ramen_name = RAMEN_NAME_MAP[idNum] ?? RAMEN_NAME_MAP[6];
//       const p = mealPeriodByKey[r.meal_period_key];

//       return {
//         ...r,
//         ramen_name,
//         meal_period: p ? { key: p.key, name: p.name, start: p.start, end: p.end }
//                        : { key: 'UNKNOWN_PERIOD', name: '未知餐期', start: null, end: null }
//       };
//     });

//     return res.json({ ok: true, data: rows });

//   } catch (err) {
//     console.error('[POST /quality/entries] error:', err);
//     return res.status(500).json({ ok: false, error: String(err?.message || err) });
//   }
// });

app.post('/store-productions', async (req, res) => {
  try {
    const body = req.body || {};

    // -----------------------------
    // 僅允許用 _id 查詢
    // -----------------------------
    const idRaw = body._id || body.id;
    if (!idRaw) {
      return res.status(400).json({ ok: false, error: '_id is required' });
    }

    let idObj;
    try {
      idObj = new ObjectId(String(idRaw));
    } catch (e) {
      return res.status(400).json({ ok: false, error: 'invalid _id format' });
    }

    // -----------------------------
    // aggregation：只查這一筆
    // -----------------------------
    const rows = await col.aggregate([
      { $match: { _id: idObj } },

      // 保留你原本的結構，不 join devices（單筆不需要）
      {
        $project: {
          _id: 0,
          time: {
            $dateToString: {
              format: '%Y-%m-%dT%H:%M:%SZ',
              date: '$event_time',
              timezone: 'UTC'
            }
          },

          // production_time_sec: 1,
          // serving_temp: 1,
          // production_time_sec: { $round: ['$production_time_sec', 0] },
          production_time_sec: {
            $let: {
              vars: {
                sec: { $toInt: { $round: ['$production_time_sec', 0] } }
              },
              in: {
                $concat: [
                  {
                    $cond: [
                      { $lt: [{ $floor: { $divide: ['$$sec', 60] } }, 10] },
                      { $concat: ['0', { $toString: { $floor: { $divide: ['$$sec', 60] } } }] },
                      { $toString: { $floor: { $divide: ['$$sec', 60] } } }
                    ]
                  },
                  ':',
                  {
                    $cond: [
                      { $lt: [{ $mod: ['$$sec', 60] }, 10] },
                      { $concat: ['0', { $toString: { $mod: ['$$sec', 60] } }] },
                      { $toString: { $mod: ['$$sec', 60] } }
                    ]
                  }
                ]
              }
            }
          },
          serving_temp: { $round: ['$serving_temp', 1] },
          status_norm: 1,

          empty_bowl_temp: { $round: [toNum('raw.Empty Bowl Temp'), 1] },
          soup_temp: { $round: [toNum('raw.Soup Temp'), 1] },
          noodle_temp: { $round: [toNum('raw.Noodle Temp'), 1] },
          // empty_bowl_temp: toNum('raw.Empty Bowl Temp'),
          // soup_temp: toNum('raw.Soup Temp'),
          // noodle_temp: toNum('raw.Noodle Temp'),

          // 拉麵名稱（先保留原始值，後面再 normalize）
          ramen_name: { $ifNull: ['$raw.ramen_name', null] },

          // toppings：raw.<key> == "1" 的項目
          toppings: {
            $filter: {
              input: TOPPING_KEYS.map(k => ({
                $cond: [
                  { $eq: [{ $toString: `$raw.${k}` }, '1'] },
                  TOPPING_LABEL_MAP[k],
                  null
                ]
              })),
              as: 'x',
              cond: { $ne: ['$$x', null] }
            }
          }
        }
      }
    ]).toArray();

    if (!rows.length) {
      return res.json({ ok: true, data: null, note: 'not found' });
    }

    // -----------------------------
    // 後處理：拉麵名稱正規化
    // -----------------------------
    const row = rows[0];
    const mappedRow = {
      ...row,
      ramen_name: normalizeRamenName(row.ramen_name)
    };

    return res.json({ ok: true, data: mappedRow });

  } catch (err) {
    console.error('[POST /quality/store-productions] error:', err);
    return res.status(500).json({ ok: false, error: String(err?.message || err) });
  }
});


// 140x140 透明 PNG（可重用）
const BLANK_140_BASE64 =
  'iVBORw0KGgoAAAANSUhEUgAAAIwAAACMCAYAAACuwEE+AAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAALEoAACxKAXd6dE0AAAFsSURBVHhe7dKxAYAwDMCw0P9/BoYeEO/S4gf8vL+BpXMLK4YhMQ'+
  'yJYUgMQ2IYEsOQGIbEMCSGITEMiWFIDENiGBLDkBiGxDAkhiExDIlhSAxDYhgSw5AYhsQwJIYhMQyJYUgMQ2IYEsOQGIbEMCSGITEMiWFIDENiGBLDkBiGxDAkhiExDIlhSAxDYhgSw5AYhsQwJIYhMQyJYUgMQ2IYEsOQGI'+
  'bEMCSGITEMiWFIDENiGBLDkBiGxDAkhiExDIlhSAxDYhgSw5AYhsQwJIYhMQyJYUgMQ2IYEsOQGIbEMCSGITEMiWFIDENiGBLDkBiGxDAkhiExDIlhSAxDYhgSw5AYhsQwJIYhMQyJYUgMQ2IYEsOQGIbEMCSGITEMiWFIDEN'+
  'iGBLDkBiGxDAkhiExDIlhSAxDYhgSw5AYhsQwJIYhMQyJYUgMQ2IYEsOQGIbEMCSGITEMiWFIDENiGBLDkBiGxDAkhiExDMHMB6a3BRS5h8v3AAAAAElFTkSuQmCC';

app.post('/store-production-images', async (req, res) => {
  try {
    const body = req.body || {};

    const idRaw = body._id || body.id;
    if (!idRaw) {
      return res.status(400).json({ ok: false, error: '_id is required' });
    }

    let idObj;
    try {
      idObj = new ObjectId(String(idRaw));
    } catch (e) {
      return res.status(400).json({ ok: false, error: 'invalid _id format' });
    }

    // ---- Mongo 相容版：從 raw 取任意 key（含空白、-）
    const rawFieldExpr = (fieldName) => ({
      $let: {
        vars: {
          kv: {
            $arrayElemAt: [
              {
                $filter: {
                  input: { $objectToArray: '$raw' },
                  as: 'it',
                  cond: { $eq: ['$$it.k', fieldName] }
                }
              },
              0
            ]
          }
        },
        in: { $ifNull: ['$$kv.v', null] }
      }
    });

    const rawTempExpr = (fieldName) => ({
      $convert: {
        input: rawFieldExpr(fieldName),
        to: 'double',
        onError: null,
        onNull: null
      }
    });

    const rows = await col.aggregate([
      { $match: { _id: idObj } },
      {
        $project: {
          _id: 0,
          time: {
            $dateToString: {
              format: '%Y-%m-%dT%H:%M:%SZ',
              date: '$event_time',
              timezone: 'UTC'
            }
          },

          // 原始圖（可能為 null）
          empty_bowl_image: '$raw.empty_bowl_image',
          soup_image: '$raw.soup_image',
          noodle_image: '$raw.noodle_image',
          serving_back_image: '$raw.serving_back_image',

          // 溫度
          empty_bowl_temp: { $round: [rawTempExpr('Empty Bowl Temp'), 1] },
          soup_temp: { $round: [rawTempExpr('Soup Temp'), 1] },
          noodle_temp: { $round: [rawTempExpr('Noodle Temp'), 1] },
          serving_back_temp: { $round: [rawTempExpr('Serving Temp - Back'), 1] },

          // 時間
          empty_bowl_time: rawFieldExpr('Empty Bowl Time'),
          soup_time: rawFieldExpr('Soup Time'),
          noodle_time: rawFieldExpr('Noodle Time'),
          serving_back_time: rawFieldExpr('Serving Time - Back')
        }
      }
    ]).toArray();

    if (!rows.length) {
      return res.json({ ok: true, data: null, note: 'not found' });
    }

    const row = rows[0];

    // ✅ Node 端補「空白 base64 圖」
    const withDefaultImage = (v) =>
      (typeof v === 'string' && v.length > 0) ? v : BLANK_140_BASE64;

    row.empty_bowl_image   = withDefaultImage(row.empty_bowl_image);
    row.soup_image         = withDefaultImage(row.soup_image);
    row.noodle_image       = withDefaultImage(row.noodle_image);
    row.serving_back_image = withDefaultImage(row.serving_back_image);

    return res.json({ ok: true, data: row });

  } catch (err) {
    console.error('[POST /quality/store-production-images] error:', err);
    return res.status(500).json({ ok: false, error: String(err?.message || err) });
  }
});


// app.post('/store-production-images', async (req, res) => {
//   try {
//     const body = req.body || {};

//     const idRaw = body._id || body.id;
//     if (!idRaw) {
//       return res.status(400).json({ ok: false, error: '_id is required' });
//     }

//     let idObj;
//     try {
//       idObj = new ObjectId(String(idRaw));
//     } catch (e) {
//       return res.status(400).json({ ok: false, error: 'invalid _id format' });
//     }

//     // ---- Mongo 相容版：從 raw 取任意 key（含空白、-）
//     // 原理：objectToArray(raw) -> [{k,v}] -> filter k==fieldName -> 取第一個 -> .v
//     const rawFieldExpr = (fieldName) => ({
//       $let: {
//         vars: {
//           kv: {
//             $arrayElemAt: [
//               {
//                 $filter: {
//                   input: { $objectToArray: '$raw' },
//                   as: 'it',
//                   cond: { $eq: ['$$it.k', fieldName] }
//                 }
//               },
//               0
//             ]
//           }
//         },
//         in: { $ifNull: ['$$kv.v', null] }
//       }
//     });

//     // ---- Temp 轉 double（避免 "88.3" 字串）
//     const rawTempExpr = (fieldName) => ({
//       $convert: {
//         input: rawFieldExpr(fieldName),
//         to: 'double',
//         onError: null,
//         onNull: null
//       }
//     });

//     const rows = await col.aggregate([
//       { $match: { _id: idObj } },
//       {
//         $project: {
//           _id: 0,
//           time: {
//             $dateToString: {
//               format: '%Y-%m-%dT%H:%M:%SZ',
//               date: '$event_time',
//               timezone: 'UTC'
//             }
//           },

//           // ✅ 固定輸出 4 張圖
//           empty_bowl_image: '$raw.empty_bowl_image',
//           soup_image: '$raw.soup_image',
//           noodle_image: '$raw.noodle_image',
//           serving_back_image: '$raw.serving_back_image',

//           // ✅ 你指定的 raw 欄位（溫度）
//           // 'empty_bowl_temp': rawTempExpr('Empty Bowl Temp'),
//           // 'soup_temp': rawTempExpr('Soup Temp'),
//           // 'noodle_temp': rawTempExpr('Noodle Temp'),
//           // 'serving_back_temp': rawTempExpr('Serving Temp - Back'),

//           empty_bowl_temp: { $round: [rawTempExpr('Empty Bowl Temp'), 1] },
//           soup_temp: { $round: [rawTempExpr('Soup Temp'), 1] },
//           noodle_temp: { $round: [rawTempExpr('Noodle Temp'), 1] },
//           serving_back_temp: { $round: [rawTempExpr('Serving Temp - Back'), 1] },

//           // ✅ 你指定的 raw 欄位（時間）— 先原樣輸出（可能是字串/數字/ISO）
//           'empty_bowl_time': rawFieldExpr('Empty Bowl Time'),
//           'soup_time': rawFieldExpr('Soup Time'),
//           'noodle_time': rawFieldExpr('Noodle Time'),
//           'serving_back_time': rawFieldExpr('Serving Time - Back')
//         }
//       }
//     ]).toArray();

//     if (!rows.length) {
//       return res.json({ ok: true, data: null, note: 'not found' });
//     }

//     return res.json({ ok: true, data: rows[0] });

//   } catch (err) {
//     console.error('[POST /quality/store-production-images] error:', err);
//     return res.status(500).json({ ok: false, error: String(err?.message || err) });
//   }
// });





//=============================================================================================================== 
/////////////////////////////////////////////////////////////////new////////////////////////////////////////////////
//=============================================================================================================== 


// ---- 健康檢查 ----
app.get('/healthz', (_req, res) => res.json({ ok: true }));

// ---- Start ----
app.listen(PORT, '0.0.0.0', () => {
  console.log(`[server] listening on 0.0.0.0:${PORT}`);
});
