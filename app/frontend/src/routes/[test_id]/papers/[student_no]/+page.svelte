<script lang="ts">
	const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;

	const { data } = $props();

	import { onMount } from 'svelte';
	import MdiPaperOff from '~icons/mdi/paper-off';
	import MdiImagePlus from '~icons/mdi/image-plus';
	import MdiCrop from '~icons/mdi/crop';

	import type { StudentAnswer } from '$lib/index.ts';
	import * as Dialog from "$lib/components/ui/dialog/index.ts";
	import { Label } from '$lib/components/ui/label/index.js';
	import ProcessImage from './ProcessImage.svelte';

	let isPageLoading: boolean = $state(true);

	let studentItems: (StudentAnswer & {label: string})[] = $state([]);

	onMount(async () => {
		// FIXME: uncomment when good to go
		try {
			const response = await fetch(
				`${apiBaseUrl}/test_instances/${data.test_id}/${data.student_no}`,
				{
					method: "GET",
					headers: {'Content-Type': 'application/json',},
				}
			);

			const result = await response.json();
			// studentItems = result.answers;

			// if (response.ok)
			// 	window.location.reload();
			// else
			// 	alert(response.status + response.statusText);
		} catch(e) {
			// alert("Failed to fetch, check your network connection and try again.\n"+e);
		} finally {
			// isPageLoading = false;
		}
	});

	// FIXME: remove (temporary!)
	setTimeout(() => {
		for (let i: number = 1; i < 5; i++)
			studentItems.push({
				answer_id: i,
				item_id: i,
				label: i.toString(),
				student_no: data.student_no,
				image_directory: i==2 ? "https://www.math-only-math.com/images/partial-fraction.jpg" : "",
				ai_evaluation: "",
				is_done_rendering: false,
			});

		isPageLoading = false;
	}, 500);
</script>


<div class="flex flex-col gap-4">
	<h2 class="font-semibold">For student {data.student_no}:</h2>
	{#if isPageLoading}
		<p>Loading...</p>
	{:else}
		{#each studentItems as item}
			<div class="card space-y-1">
				<Label for={item.label} class="flex flex-row justify-between">
					{item.label}
					<span class="flex flex-row space-x-1">
						<Dialog.Root>
							<Dialog.Trigger class="button-secondary border-none">
								<MdiImagePlus />
							</Dialog.Trigger>
							<ProcessImage test_id={data.test_id} student_no={data.student_no} studentItem={item}/>
						</Dialog.Root>
						<a href={`/${data.test_id}/papers/${data.student_no}/manual`} class="button-secondary border-none">
							<MdiCrop />
						</a>
					</span>
				</Label>
				<div class="flex justify-center items-center">
					{#if item.image_directory == ""}
						<MdiPaperOff class="h-12 w-full py-2 bg-muted text-muted-foreground border" />
					{:else}
						<img src={item.image_directory} class="size-fill max-h-[25vh]" alt={item.label}/>
					{/if}
				</div>
			</div>
		{/each}
	{/if}
</div>
