/**
 * Best-effort in-memory rate limiter.
 *
 * NOTE: This is an in-memory rate limiter scoped to a single server instance.
 * On serverless platforms like Vercel, requests may be distributed across multiple
 * stateless instances. This provides reasonable protection against casual local spam,
 * but does NOT guarantee a global multi-region quota.
 */

interface RateLimitRecord {
  timestamps: number[];
}

const windowMs = 60 * 1000; // 1 minute window
const rateLimitMap = new Map<string, RateLimitRecord>();

// Periodic garbage collection to keep memory bounded
let lastCleanup = Date.now();

function cleanupStaleEntries() {
  const now = Date.now();
  if (now - lastCleanup < 60000) return;
  lastCleanup = now;

  const threshold = now - windowMs;
  rateLimitMap.forEach((record, key) => {
    record.timestamps = record.timestamps.filter((ts) => ts > threshold);
    if (record.timestamps.length === 0) {
      rateLimitMap.delete(key);
    }
  });
}

export function checkRateLimit(
  clientIp: string,
  maxRequestsPerMinute: number = 20
): { allowed: boolean; remaining: number; resetSeconds: number } {
  cleanupStaleEntries();

  const now = Date.now();
  const threshold = now - windowMs;

  let record = rateLimitMap.get(clientIp);
  if (!record) {
    record = { timestamps: [] };
    rateLimitMap.set(clientIp, record);
  }

  // Filter timestamps within current window
  record.timestamps = record.timestamps.filter((ts) => ts > threshold);

  if (record.timestamps.length >= maxRequestsPerMinute) {
    const oldest = record.timestamps[0] || now;
    const resetSeconds = Math.max(1, Math.ceil((oldest + windowMs - now) / 1000));
    return {
      allowed: false,
      remaining: 0,
      resetSeconds,
    };
  }

  record.timestamps.push(now);
  return {
    allowed: true,
    remaining: Math.max(0, maxRequestsPerMinute - record.timestamps.length),
    resetSeconds: 60,
  };
}
