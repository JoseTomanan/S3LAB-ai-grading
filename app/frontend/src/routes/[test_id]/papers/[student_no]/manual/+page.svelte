<script lang="ts">
	const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;

	const { data } = $props();
</script>

<div class="flex flex-col space-y-2">
	<h2 class="font-bold">Manually crop image</h2>
	<h6>{data.test_id} &middot; {data.student_no} &middot; ...</h6>
	<!-- TODO: all -->
</div>