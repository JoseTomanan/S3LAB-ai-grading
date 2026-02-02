<script lang="ts">
	let { data } = $props();
		
	import { onMount } from "svelte";
	import MdiPaperAdd from '~icons/mdi/paper-add';

	import ProcessImage from "$lib/components/dialogs/ProcessImage.svelte";
	import * as Dialog from "$lib/components/ui/dialog/index.ts";
	import type { Student, TestItemHeader } from "$lib/types/types.ts";

	let isPageLoading: boolean = $state(true);
	let students: Student[] = $state([]);
	let itemHeaders: TestItemHeader[] = $state([]);

	onMount(async () => {
		// TODO: finish; revert console.log() to alert() and isPageLoading when done.
		try {
			const response = await fetch(``, {});
		} catch(e) {
			console.log("Failed to fetch, check your network connection and try again.");
		} finally {
		}
	});

	// FIXME: remove (temporary!)
	setTimeout(() => {
		students = [
			{student_no: "2021-00000", name: "ABALOS, Benhur E.", section: "3-Acacia"},
			{student_no: "2021-00001", name: "ANDAL, David V.", section: "3-Acacia"}
		];
		itemHeaders = [ {item_id: 1, label: '1a'}, {item_id: 2, label: '1b'} ];
		isPageLoading = false;
	}, 500);
</script>

<div class="flex flex-col gap-4">
	{#if isPageLoading}
		<p>Loading...</p>
	{:else}
		{#each students as student}
			<Dialog.Root>
				<Dialog.Trigger>
					<MdiPaperAdd class="size-8"/>
				</Dialog.Trigger>
				<ProcessImage {student} {itemHeaders}/>
			</Dialog.Root>
		{/each}
	{/if}
</div>