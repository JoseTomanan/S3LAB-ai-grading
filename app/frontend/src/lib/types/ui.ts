import type { GetSpecificEvaluationResponse } from "./schemas.ts";
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

export type FileRecord = {
  file: File;
  name: string;
  url: string;
  statusCode: number;
  statusDetail?: string;
};

// ================ CONTEXT ================
export interface TestItemsContext {
	items: TestItem[];
	isLoading: boolean;
}

