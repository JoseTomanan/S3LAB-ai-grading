import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
	return twMerge(clsx(inputs));
}

export function dataUrlToFile(dataUrl: string, filename: string) {
		// Separate out the mime component and the base64 data
		const arr = dataUrl.split(',');
		const mime = arr[0].match(/:(.*?);/)![1];
		const bstr = atob(arr[arr.length - 1]); // Decode base64 to binary string
		let n = bstr.length;
		const u8arr = new Uint8Array(n); // Create a Uint8Array

		// Write the bytes of the string to the Uint8Array
		while (n--)
				u8arr[n] = bstr.charCodeAt(n);

		// Create a Blob and then a File object
		const blob = new Blob([u8arr], { type: mime });
		return new File([blob], filename, { type: mime });
}

export function filesToFileList(fileArray: File[]) {
		console.log("--> It reached here lol")
    return fileArray as unknown as FileList;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type WithoutChild<T> = T extends { child?: any } ? Omit<T, "child"> : T;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type WithoutChildren<T> = T extends { children?: any } ? Omit<T, "children"> : T;
export type WithoutChildrenOrChild<T> = WithoutChildren<WithoutChild<T>>;
export type WithElementRef<T, U extends HTMLElement = HTMLElement> = T & { ref?: U | null };
