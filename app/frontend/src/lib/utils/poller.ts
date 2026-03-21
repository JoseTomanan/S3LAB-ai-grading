/**
 * Creates a polling utility that calls `fn` at a fixed interval.
 * Polling stops when `fn` returns `true` (done) or `maxRetries` is reached.
 */
export function createPoller(
	fn: () => Promise<boolean>,
	intervalMs = 5000,
	maxRetries = 60
) {
	let interval: ReturnType<typeof setInterval> | null = null;
	let retries = 0;

	function start() {
		if (interval) return;
		retries = 0;
		interval = setInterval(async () => {
			retries++;
			if (retries >= maxRetries) {
				stop();
				return;
			}
			try {
				const done = await fn();
				if (done) stop();
			} catch {
				// silently continue polling
			}
		}, intervalMs);
	}

	function stop() {
		if (interval) {
			clearInterval(interval);
			interval = null;
		}
	}

	return {
		start,
		stop,
		get active() {
			return interval !== null;
		}
	};
}
