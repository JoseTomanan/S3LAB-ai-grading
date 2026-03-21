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

/**
 * Typed fetch wrapper for JSON API calls.
 * Throws ApiError on non-OK responses.
 * Returns undefined for 204 No Content.
 */
export async function api<T>(
	url: string,
	options?: RequestInit
): Promise<T> {
	const response = await fetch(url, {
		headers: { 'Content-Type': 'application/json', ...options?.headers },
		...options
	});

	if (response.status === 204) return undefined as T;

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

	return response.json();
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

	return response.json();
}
