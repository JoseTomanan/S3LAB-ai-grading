/**
 * Creates a polling utility that calls `fn` at a fixed interval.
 * Polling stops when `fn` returns `true` (done) or `maxRetries` is reached.
 * Throws an error if polling function fails.
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
				const timeoutError = new Error(
					`Polling timed out after ${maxRetries} retries (${(maxRetries * intervalMs) / 1000}s)`
				);
				timeoutError.name = 'PollingTimeoutError';
				throw timeoutError;
			}
			try {
				const done = await fn();
				if (done) stop();
			} catch (error) {
				stop();
				
				// Enhance error message based on error type
				if (error instanceof TypeError && error.message.includes('fetch')) {
					const networkError = new Error(
						'Network error during polling: Unable to reach server. Please check your connection.'
					);
					networkError.name = 'PollingNetworkError';
					throw networkError;
				}
				
				// Handle HTTP errors
				if (error instanceof Response || (error as any)?.status) {
					const status = (error as any).status;
					const serverError = new Error(
						`Server error during polling: ${status >= 500 ? 'Server is experiencing issues' : `Request failed with status ${status}`}`
					);
					serverError.name = 'PollingServerError';
					throw serverError;
				}
				
				// Re-throw with context if it's already an error
				if (error instanceof Error) {
					error.message = `Polling failed: ${error.message}`;
					throw error;
				}
				
				// Unknown error type
				throw new Error(`Polling failed with unexpected error: ${String(error)}`);
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
