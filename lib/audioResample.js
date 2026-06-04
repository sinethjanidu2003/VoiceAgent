/** Linear resample between 16 kHz and 24 kHz PCM16 mono. */

export function resample16kTo24k(pcmBuffer) {
  const inSamples = pcmBuffer.length / 2;
  if (!inSamples) return Buffer.alloc(0);

  const input = new Int16Array(
    pcmBuffer.buffer,
    pcmBuffer.byteOffset,
    inSamples
  );
  const outLen = Math.floor((inSamples * 24000) / 16000);
  const output = new Int16Array(outLen);

  for (let i = 0; i < outLen; i++) {
    const srcPos = (i * 16000) / 24000;
    const idx = Math.floor(srcPos);
    const frac = srcPos - idx;
    const s0 = input[idx] ?? 0;
    const s1 = input[Math.min(idx + 1, inSamples - 1)] ?? 0;
    output[i] = Math.round(s0 + frac * (s1 - s0));
  }

  return Buffer.from(output.buffer, output.byteOffset, output.byteLength);
}

export function resample24kTo16k(pcmBuffer) {
  const inSamples = pcmBuffer.length / 2;
  if (!inSamples) return Buffer.alloc(0);

  const input = new Int16Array(
    pcmBuffer.buffer,
    pcmBuffer.byteOffset,
    inSamples
  );
  const outLen = Math.floor((inSamples * 16000) / 24000);
  const output = new Int16Array(outLen);

  for (let i = 0; i < outLen; i++) {
    const srcPos = (i * 24000) / 16000;
    const idx = Math.floor(srcPos);
    const frac = srcPos - idx;
    const s0 = input[idx] ?? 0;
    const s1 = input[Math.min(idx + 1, inSamples - 1)] ?? 0;
    output[i] = Math.round(s0 + frac * (s1 - s0));
  }

  return Buffer.from(output.buffer, output.byteOffset, output.byteLength);
}
