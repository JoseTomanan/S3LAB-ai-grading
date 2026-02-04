export interface Point {
	x: number;
	y: number;
}

export interface CropData {
	coordinates: Point[];
	originalWidth: number;
	originalHeight: number;
}