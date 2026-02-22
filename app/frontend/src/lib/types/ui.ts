import type { TestItem } from "./types.ts";

export interface Point {
	x: number;
	y: number;
}

export interface CropData {
	coordinates: Point[];
	originalWidth: number;
	originalHeight: number;
}

// ============ CONTEXT ============
export interface TestItemsContext {
	items: TestItem[];
	isLoading: boolean;
}
