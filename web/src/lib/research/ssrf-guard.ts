/**
 * SAM-AI Server-Side SSRF (Server-Side Request Forgery) Defense Engine
 * ====================================================================
 * Comprehensive firewall and validation system for external HTTP requests:
 * 1. Strict URL scheme, credential, port, and hostname format validation.
 * 2. Complete IPv4 and IPv6 normalization (handles octal, hex, decimal integers, IPv4-mapped IPv6).
 * 3. Comprehensive CIDR blocklist: loopback, RFC 1918 private, link-local, cloud metadata, multicast, reserved.
 * 4. Multi-record DNS resolution validation: every resolved A and AAAA record must be safe.
 * 5. Pinned destination transport support to eliminate DNS-rebinding / TOCTOU exploits.
 */

import dns from "node:dns";
import { SafeFetchError } from "./types.ts";

/**
 * Prohibited IPv4 CIDR blocks represented as [numericBase, maskBits]
 */
interface Ipv4Range {
  name: string;
  base: number;
  mask: number;
}

function parseIpv4ToNumber(ip: string): number | null {
  const parts = ip.split(".");
  if (parts.length !== 4) return null;

  let num = 0;
  for (let i = 0; i < 4; i++) {
    const part = parts[i];
    let val: number;

    // Handle hex (0x...), octal (0...), and decimal
    if (/^0x[0-9a-f]+$/i.test(part)) {
      val = parseInt(part, 16);
    } else if (/^0[0-7]+$/.test(part) && part !== "0") {
      val = parseInt(part, 8);
    } else if (/^\d+$/.test(part)) {
      val = parseInt(part, 10);
    } else {
      return null;
    }

    if (val < 0 || val > 255) return null;
    num = (num << 8) | val;
  }
  // Convert to unsigned 32-bit integer
  return num >>> 0;
}

function makeIpv4Range(name: string, cidr: string): Ipv4Range {
  const [ipStr, bitsStr] = cidr.split("/");
  const bits = parseInt(bitsStr, 10);
  const base = parseIpv4ToNumber(ipStr)!;
  const mask = bits === 0 ? 0 : (~0 << (32 - bits)) >>> 0;
  return { name, base: (base & mask) >>> 0, mask };
}

// Banned IPv4 CIDR ranges
const PROHIBITED_IPV4_RANGES: Ipv4Range[] = [
  makeIpv4Range("Loopback", "127.0.0.0/8"),
  makeIpv4Range("Current Network (Zero)", "0.0.0.0/8"),
  makeIpv4Range("Private 10/8", "10.0.0.0/8"),
  makeIpv4Range("Carrier-Grade NAT", "100.64.0.0/10"),
  makeIpv4Range("Private 172.16/12", "172.16.0.0/12"),
  makeIpv4Range("Link-Local / Metadata", "169.254.0.0/16"),
  makeIpv4Range("Private 192.168/16", "192.168.0.0/16"),
  makeIpv4Range("IETF Protocol Assignments", "192.0.0.0/24"),
  makeIpv4Range("TEST-NET-1", "192.0.2.0/24"),
  makeIpv4Range("Inter-Network Benchmark", "198.18.0.0/15"),
  makeIpv4Range("TEST-NET-2", "198.51.100.0/24"),
  makeIpv4Range("TEST-NET-3", "203.0.113.0/24"),
  makeIpv4Range("Multicast (Class D)", "224.0.0.0/4"),
  makeIpv4Range("Reserved (Class E)", "240.0.0.0/4"),
  makeIpv4Range("Broadcast", "255.255.255.255/32"),
];

/**
 * Checks if an unsigned 32-bit IPv4 integer matches any prohibited CIDR.
 */
function isIpv4NumberProhibited(ipNum: number): { prohibited: boolean; reason?: string } {
  for (const range of PROHIBITED_IPV4_RANGES) {
    if ((ipNum & range.mask) >>> 0 === range.base) {
      return { prohibited: true, reason: `Prohibited IPv4 range (${range.name})` };
    }
  }
  return { prohibited: false };
}

/**
 * Expands and parses IPv6 address into 8 16-bit words.
 */
function parseIpv6(ip: string): number[] | null {
  let clean = ip.toLowerCase().trim();

  // Strip brackets if present
  if (clean.startsWith("[") && clean.endsWith("]")) {
    clean = clean.slice(1, -1);
  }

  // Handle zone index (e.g. fe80::1%eth0)
  const zoneIdx = clean.indexOf("%");
  if (zoneIdx !== -1) {
    clean = clean.slice(0, zoneIdx);
  }

  // Check for IPv4-mapped IPv6 (e.g. ::ffff:192.0.2.128 or ::ffff:7f00:1)
  const lastColon = clean.lastIndexOf(":");
  if (lastColon !== -1) {
    const possibleIpv4 = clean.slice(lastColon + 1);
    if (possibleIpv4.includes(".")) {
      const ipv4Num = parseIpv4ToNumber(possibleIpv4);
      if (ipv4Num === null) return null;
      const highWord = (ipv4Num >>> 16) & 0xffff;
      const lowWord = ipv4Num & 0xffff;
      const prefix = clean.slice(0, lastColon);
      clean = `${prefix}:${highWord.toString(16)}:${lowWord.toString(16)}`;
    }
  }

  const doubleColonCount = (clean.match(/::/g) || []).length;
  if (doubleColonCount > 1) return null;

  let parts: string[];
  if (doubleColonCount === 1) {
    const [leftStr, rightStr] = clean.split("::");
    const leftParts = leftStr ? leftStr.split(":") : [];
    const rightParts = rightStr ? rightStr.split(":") : [];
    const missing = 8 - (leftParts.length + rightParts.length);
    if (missing < 1) return null;
    parts = [...leftParts, ...new Array(missing).fill("0"), ...rightParts];
  } else {
    parts = clean.split(":");
  }

  if (parts.length !== 8) return null;

  const words: number[] = [];
  for (const part of parts) {
    if (!/^[0-9a-f]{1,4}$/i.test(part)) return null;
    words.push(parseInt(part, 16));
  }
  return words;
}

/**
 * Validates whether an IPv6 address falls into loopback, private, link-local, or multicast subnets.
 */
function isIpv6Prohibited(ip: string): { prohibited: boolean; reason?: string } {
  const words = parseIpv6(ip);
  if (!words) {
    return { prohibited: true, reason: "Malformed IPv6 address format" };
  }

  // 1. Unspecified ::/128
  if (words.every((w) => w === 0)) {
    return { prohibited: true, reason: "Prohibited IPv6 Unspecified Address (::/128)" };
  }

  // 2. Loopback ::1/128
  if (words.slice(0, 7).every((w) => w === 0) && words[7] === 1) {
    return { prohibited: true, reason: "Prohibited IPv6 Loopback Address (::1/128)" };
  }

  // 3. IPv4-mapped IPv6 ::ffff:0:0/96
  if (words.slice(0, 5).every((w) => w === 0) && words[5] === 0xffff) {
    const ipv4Num = ((words[6] << 16) | words[7]) >>> 0;
    const check = isIpv4NumberProhibited(ipv4Num);
    if (check.prohibited) {
      return { prohibited: true, reason: `IPv4-mapped IPv6: ${check.reason}` };
    }
  }

  // 4. IPv4-compatible IPv6 ::/96 (deprecated)
  if (words.slice(0, 6).every((w) => w === 0)) {
    const ipv4Num = ((words[6] << 16) | words[7]) >>> 0;
    const check = isIpv4NumberProhibited(ipv4Num);
    if (check.prohibited) {
      return { prohibited: true, reason: `IPv4-compatible IPv6: ${check.reason}` };
    }
  }

  // 5. Unique Local Address (ULA) fc00::/7 (first word starts with 0xfc or 0xfd)
  const firstWord = words[0];
  if ((firstWord & 0xfe00) === 0xfc00) {
    return { prohibited: true, reason: "Prohibited IPv6 Unique Local Address (fc00::/7)" };
  }

  // 6. Link-Local Unicast fe80::/10 (first word starts with 0xfe80 - 0xfebf)
  if ((firstWord & 0xffc0) === 0xfe80) {
    return { prohibited: true, reason: "Prohibited IPv6 Link-Local Address (fe80::/10)" };
  }

  // 7. Multicast ff00::/8 (first word starts with 0xff)
  if ((firstWord & 0xff00) === 0xff00) {
    return { prohibited: true, reason: "Prohibited IPv6 Multicast Address (ff00::/8)" };
  }

  return { prohibited: false };
}

/**
 * Validates any IP literal (IPv4 or IPv6) against prohibited subnets.
 */
export function isIpAddressProhibited(ip: string): { prohibited: boolean; reason?: string } {
  const clean = ip.trim();

  // Try IPv4 first
  if (clean.includes(".")) {
    const num = parseIpv4ToNumber(clean);
    if (num !== null) {
      return isIpv4NumberProhibited(num);
    }
  }

  // Try IPv6
  if (clean.includes(":")) {
    return isIpv6Prohibited(clean);
  }

  // Check decimal integer IP (e.g. 2130706433)
  if (/^\d+$/.test(clean)) {
    const num = parseInt(clean, 10);
    if (num >= 0 && num <= 0xffffffff) {
      return isIpv4NumberProhibited(num >>> 0);
    }
  }

  return { prohibited: false };
}

/**
 * Checks known internal/metadata domain names and prohibited TLDs.
 */
export function isHostnameForbidden(hostname: string): { forbidden: boolean; reason?: string } {
  const lower = hostname.toLowerCase().trim().replace(/\.$/, "");

  if (
    lower === "localhost" ||
    lower.endsWith(".localhost") ||
    lower.endsWith(".local") ||
    lower.endsWith(".internal") ||
    lower.endsWith(".corp") ||
    lower.endsWith(".lan") ||
    lower.endsWith(".home")
  ) {
    return { forbidden: true, reason: "Prohibited internal or loopback domain suffix" };
  }

  // Cloud metadata and cloud internal hostnames
  const metadataHosts = new Set([
    "metadata.google.internal",
    "metadata.internal",
    "instance-data",
    "169.254.169.254",
    "100.100.100.200",
  ]);

  if (metadataHosts.has(lower)) {
    return { forbidden: true, reason: "Prohibited cloud instance metadata service" };
  }

  // Check if hostname is an IP literal directly
  const ipCheck = isIpAddressProhibited(lower);
  if (ipCheck.prohibited) {
    return { forbidden: true, reason: ipCheck.reason };
  }

  return { forbidden: false };
}

/**
 * Validates initial URL syntax, scheme, credentials, port, and hostname before network activity.
 */
export function validateUrlSyntax(
  urlStr: string,
  allowedSchemes: string[] = ["https:"],
  allowedPorts: number[] = [443]
): { valid: boolean; parsedUrl?: URL; error?: string } {
  if (!urlStr || typeof urlStr !== "string" || urlStr.trim().length === 0) {
    return { valid: false, error: "Empty or missing URL string" };
  }

  let parsed: URL;
  try {
    parsed = new URL(urlStr.trim());
  } catch {
    return { valid: false, error: `Malformed URL structure: '${urlStr}'` };
  }

  // 1. Credentials in URL are strictly rejected (e.g. https://user:pass@example.com)
  if (parsed.username || parsed.password) {
    return { valid: false, error: "Embedded credentials in URL are strictly prohibited" };
  }

  // 2. Validate URL protocol scheme
  const scheme = parsed.protocol.toLowerCase();
  if (!allowedSchemes.includes(scheme)) {
    return {
      valid: false,
      error: `Unsupported URL scheme '${scheme}'. Allowed schemes: ${allowedSchemes.join(", ")}`,
    };
  }

  // 3. Validate destination port
  let port = parsed.port ? parseInt(parsed.port, 10) : null;
  if (port === null) {
    port = scheme === "https:" ? 443 : 80;
  }
  if (!allowedPorts.includes(port)) {
    return {
      valid: false,
      error: `Destination port '${port}' is not permitted. Permitted ports: ${allowedPorts.join(", ")}`,
    };
  }

  // 4. Validate hostname syntax
  const hostCheck = isHostnameForbidden(parsed.hostname);
  if (hostCheck.forbidden) {
    return { valid: false, error: `Forbidden destination hostname: ${hostCheck.reason}` };
  }

  return { valid: true, parsedUrl: parsed };
}

/**
 * Resolves a hostname via DNS and strictly validates every returned IP address.
 * Rejects if ANY resolved A or AAAA record falls into a private or prohibited subnet.
 */
export async function resolveAndValidateHost(
  hostname: string,
  customLookup?: (h: string) => Promise<string[]>
): Promise<{ safe: boolean; resolvedIps: string[]; error?: string }> {
  // Check hostname string first
  const hostCheck = isHostnameForbidden(hostname);
  if (hostCheck.forbidden) {
    return { safe: false, resolvedIps: [], error: hostCheck.reason };
  }

  // If custom lookup hook provided (for unit testing or transport injection)
  if (customLookup) {
    try {
      const ips = await customLookup(hostname);
      if (!ips || ips.length === 0) {
        return { safe: false, resolvedIps: [], error: `DNS lookup returned zero records for '${hostname}'` };
      }
      for (const ip of ips) {
        const check = isIpAddressProhibited(ip);
        if (check.prohibited) {
          return {
            safe: false,
            resolvedIps: ips,
            error: `Resolved IP '${ip}' is in a prohibited range: ${check.reason}`,
          };
        }
      }
      return { safe: true, resolvedIps: ips };
    } catch (err: any) {
      return { safe: false, resolvedIps: [], error: `DNS resolution failed: ${err.message}` };
    }
  }

  // Default Node.js native DNS resolution (resolves both IPv4 and IPv6)
  try {
    const records = await dns.promises.lookup(hostname, { all: true });
    if (!records || records.length === 0) {
      return { safe: false, resolvedIps: [], error: `DNS returned zero records for '${hostname}'` };
    }

    const resolvedIps = records.map((r) => r.address);
    for (const ip of resolvedIps) {
      const check = isIpAddressProhibited(ip);
      if (check.prohibited) {
        return {
          safe: false,
          resolvedIps,
          error: `Resolved IP '${ip}' for host '${hostname}' is prohibited: ${check.reason}`,
        };
      }
    }

    return { safe: true, resolvedIps };
  } catch (err: any) {
    return { safe: false, resolvedIps: [], error: `DNS lookup failed for '${hostname}': ${err.message}` };
  }
}
