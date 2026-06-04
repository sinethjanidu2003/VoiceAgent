import { createStreamingSession } from "./streamingSession.js";

/** @deprecated Use createStreamingSession — kept as alias for index.js */
export function createStreamingBridge(options) {
  return Promise.resolve(createStreamingSession(options));
}

export { createStreamingSession };
