<script lang="ts">
	let { isOpen = $bindable(false) } = $props();

	import { onMount, onDestroy } from 'svelte';
	import MdiUpload from "~icons/mdi/upload";
	import MdiRefresh from "~icons/mdi/refresh";

	import fabric from 'fabric'; 
	import * as Dialog from "$lib/components/ui/dialog/index.ts";
	import { Button } from "$lib/components/ui/button/index.ts";

	let canvasEl: HTMLCanvasElement;
	let fabricCanvas: fabric.Canvas | null = null;
	let fileInput: HTMLInputElement;
	let hasImage = $state(false);

	// Initialize Canvas when Dialog opens
	$effect(() => {
		if (isOpen && canvasEl && !fabricCanvas) {
			fabricCanvas = new fabric.Canvas(canvasEl, {
						width: 500,
						height: 400,
						backgroundColor: '#f4f4f5', // Matches Shadcn 'muted'
						selection: false // Disable group selection
						});
		} else if (!isOpen && fabricCanvas) {
			// Cleanup when closed
			fabricCanvas.dispose();
			fabricCanvas = null;
			hasImage = false;
		}
	});

	const handleUpload = (e: Event) => {
		const file = (e.target as HTMLInputElement).files?.[0];
		if (!file || !fabricCanvas) return;

		const url = URL.createObjectURL(file);

		// Fabric's robust image loader
		// @ts-ignore
		fabric.Image.fromURL(url, (img) => {
			if (!fabricCanvas) return;
			
			fabricCanvas.clear();
			
			// 1. Scale Image to fit the 500x400 Canvas
			const scale = Math.min(
						500 / (img.width || 1), 
						400 / (img.height || 1)
					);
					
			img.set({
						scaleX: scale,
						scaleY: scale,
						left: 250, // Center X
						top: 200,  // Center Y
						originX: 'center',
						originY: 'center',
						selectable: false, // User can't move the base image
						evented: false
					});

			fabricCanvas.add(img);
			addCropBox(); // Add the drag handles
			hasImage = true; // Update state
			fabricCanvas.renderAll();
		});
	};

	const addCropBox = () => {
		if (!fabricCanvas) return;

		// Define 4 corner points (Trapezoid shape default)
		const points = [
			{ x: 150, y: 100 }, // TL
			{ x: 350, y: 100 }, // TR
			{ x: 400, y: 300 }, // BR
			{ x: 100, y: 300 }  // BL
		];

		// The Visual Polygon (Semi-transparent overlay)
		const polygon = new fabric.Polygon(points, {
					fill: 'rgba(255, 255, 255, 0.2)',
					stroke: 'white',
					strokeWidth: 2,
					selectable: false,
					objectCaching: false,
					name: 'crop-polygon'
				} as any);
		fabricCanvas.add(polygon);

		// Draggable Handles
		points.forEach((point, index) => {
			const circle = new fabric.Circle({
						radius: 12,
						fill: '#3b82f6', // Tailwind blue-500
						stroke: '#ffffff',
						strokeWidth: 2,
						left: point.x,
						top: point.y,
						originX: 'center',
						originY: 'center',
						hasControls: false,
						hasBorders: false,
						name: `handle-${index}`
					});

			// Update Polygon when handle moves
			circle.on('moving', () => {
						const p = polygon.points![index];
						p.x = circle.left!;
						p.y = circle.top!;
						// Force re-render of the polygon shape
						fabricCanvas?.requestRenderAll(); 
					});

			// @ts-ignore
			fabricCanvas.add(circle);
		});
	};

	const confirmCrop = () => {
		if (!fabricCanvas) return;
		
		// Find our polygon to get the final points
		// @ts-ignore
		const polygon = fabricCanvas.getObjects().find(o => o.name === 'crop-polygon') as fabric.Polygon;
		
		if (polygon && polygon.points) {
			console.log("FINAL COORDINATES:", polygon.points);
			// Here you can close the dialog or send to API
			isOpen = false;
		}
	};
</script>


<Dialog.Root bind:open={isOpen}>
	<Dialog.Trigger>
		Open dialog
	</Dialog.Trigger>
	<Dialog.Content class="sm:max-w-[600px]">
		<Dialog.Header>
			<Dialog.Title>Irregular Crop</Dialog.Title>
			<Dialog.Description>
				Upload an image and drag the blue handles to define the shape.
			</Dialog.Description>
		</Dialog.Header>

		<div class="relative w-full h-[400px] rounded-md overflow-hidden border border-border flex items-center justify-center bg-muted">
			<canvas bind:this={canvasEl}></canvas>
			
			{#if !hasImage}
				<div class="absolute inset-0 flex flex-col items-center justify-center text-muted-foreground pointer-events-none z-10">
					<MdiUpload class="size-8 mb-2 opacity-50" />
					<span class="text-sm">No image loaded</span>
				</div>
			{/if}
		</div>

		<input 
			bind:this={fileInput} 
			type="file" 
			accept="image/*" 
			class="hidden" 
			onchange={handleUpload} 
		/>

		<Dialog.Footer>
			<Button variant="outline" onclick={() => fileInput.click()}>
				{#if hasImage}
					<MdiRefresh class="mr-2 h-4 w-4" /> Change Image
				{:else}
					<MdiUpload class="mr-2 h-4 w-4" /> Upload
				{/if}
			</Button>
			
			<Button disabled={!hasImage} onclick={confirmCrop}>
				Crop & Save
			</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>