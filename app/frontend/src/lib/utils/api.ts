export class ApiError extends Error {
	status: number;
	statusText: string;
	detail?: string;

	constructor(status: number, statusText: string, detail?: string) {
		super(detail ?? `${status} ${statusText}`);
		this.status = status;
		this.statusText = statusText;
		this.detail = detail;
	}
}

/* Health check function to verify API availability. */
export async function isApiHealthy(): Promise<boolean> {
  return fetch('/api/health')
    .then(res => res.ok)
    .catch(() => false);
}

/**
 * Typed fetch wrapper for JSON API calls.
 * Throws ApiError on non-OK responses.
 * Returns undefined for responses with no body (e.g. 202, 204).
 */
export async function api<T>(
	url: string,
	options?: RequestInit,
	customFetch?: typeof fetch
): Promise<T> {
	const fetchFn = customFetch ?? fetch;
	const response = await fetchFn(url, {
		headers: { 'Content-Type': 'application/json', ...options?.headers },
		...options
	});

	if (!response.ok) {
		let detail: string | undefined;
		try {
			const body = await response.json();
			detail = body.detail;
		} catch {
			// no JSON body
		}
		throw new ApiError(response.status, response.statusText, detail);
	}

	const text = await response.text();
	if (!text) return undefined as T;
	return JSON.parse(text);
}

/**
 * Fetch wrapper for FormData uploads (no Content-Type header — browser sets boundary).
 * Throws ApiError on non-OK responses.
 */
export async function apiForm<T>(
	url: string,
	body: FormData,
	method = 'POST'
): Promise<T> {
	const response = await fetch(url, { method, body });

	if (!response.ok) {
		let detail: string | undefined;
		try {
			const jsonBody = await response.json();
			detail = jsonBody.detail;
		} catch {
			// no JSON body
		}
		throw new ApiError(response.status, response.statusText, detail);
	}

	const text = await response.text();
	if (!text) return undefined as T;
	return JSON.parse(text);
}
