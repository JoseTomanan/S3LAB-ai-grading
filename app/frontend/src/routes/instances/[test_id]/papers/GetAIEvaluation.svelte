<script lang="ts">
	const { test_id, student_no } = $props();

	import { API_BASE_URL } from '$lib/constants.ts';
	import { onMount } from "svelte";

	import * as Dialog from "$lib/components/ui/dialog/index.ts";

	let isPageLoading = $state(false);

	onMount(async () => {
		try {
		} catch (e) {
		} finally {
		}
	});
</script>


<Dialog.Content>
	<Dialog.Header>
		<Dialog.Title>AI evaluation results</Dialog.Title>
		<Dialog.Description>{test_id} &middot; {student_no}</Dialog.Description>
	</Dialog.Header>
	<!-- TODO: content -->
</Dialog.Content>