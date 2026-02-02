<script lang="ts">
	const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;

	import MdiPlus from '~icons/mdi/plus';

	import type { Student, TestItemHeader } from '$lib/types/types.ts';
	import * as Dialog from '$lib/components/ui/dialog/index.js';
	import { Label } from '$lib/components/ui/label/index.js';

	let { student, itemHeaders }: { student: Student, itemHeaders: TestItemHeader[] } = $props();
</script>

<Dialog.Content>
	<Dialog.Title>For {student.name}</Dialog.Title>
	{#each itemHeaders as item}
		<span class="flex flex-row gap-4">
			<Label for={student.student_no}>{item.label}</Label>
			<button class="button-secondary">
				<MdiPlus />
			</button>
		</span>
		<!-- TODO: forms, image cropping interface, etc. -->
	{/each}
</Dialog.Content>