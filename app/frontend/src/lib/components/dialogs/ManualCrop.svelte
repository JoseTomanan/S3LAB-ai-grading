<script lang="ts">
	const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;
	
	let manualCropPoints: {x: number, y: number}[] = $state([
				{x: 0, y: 0}, {x: 25, y: 0}, {x: 25, y: 25}, {x: 0, y: 25}, 
				]);

	async function recrop() {
		// TODO: function
	}
</script>


