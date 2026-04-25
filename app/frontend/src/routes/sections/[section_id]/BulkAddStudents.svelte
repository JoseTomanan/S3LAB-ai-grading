<script lang="ts">
	import { API_URL } from '$lib/constants.ts';

	let { section_id, isBulkDialogOpen = $bindable() } = $props();

	import { invalidateAll } from '$app/navigation';
	import { api, ApiError } from '$lib/utils/api.ts';
	import toast from 'svelte-5-french-toast';
	import * as Dialog from '$lib/components/ui/dialog/index.ts';
	import Button from '$lib/components/CustomButton/button.svelte';
	import { Textarea } from '$lib/components/ui/textarea/index.ts';
	import { Label } from '$lib/components/ui/label/index.ts';
	import MdiCheck from '~icons/mdi/check';
	import MdiClose from '~icons/mdi/close';
	import MdiAlertCircle from '~icons/mdi/alert-circle';

	type RowResult = {
		lineNo: number;
		student_no: string;
		name: string;
		status: 'success' | 'error' | 'invalid';
		detail?: string;
	};

	let formText: string = $state('');
	let isLoading: boolean = $state(false);
	let results: RowResult[] | null = $state(null);

	function parseLines(text: string): RowResult[] {
		return text
			.split('\n')
			.map((l) => l.trim())
			.filter(Boolean)
			.map((line, i) => {
				const lineNo = i + 1;
				const idx = line.indexOf(',');
				if (idx === -1) {
					return { lineNo, student_no: line, name: '', status: 'invalid' as const, detail: 'Missing comma separator' };
				}
				const student_no = line.slice(0, idx).trim();
				const name = line.slice(idx + 1).trim();
				if (!student_no || !name) {
					return { lineNo, student_no, name, status: 'invalid' as const, detail: 'Empty student number or name' };
				}
				return { lineNo, student_no, name, status: 'success' as const };
			});
	}

	async function handleBulkAdd() {
		const parsed = parseLines(formText);
		if (parsed.length === 0) {
			toast('No rows to add');
			return;
		}

		isLoading = true;
		const valid = parsed.filter((r) => r.status !== 'invalid');
		const settled = await Promise.all(
			valid.map((row) =>
				api(`${API_URL}/api/students`, {
					method: 'POST',
					body: JSON.stringify({ student_no: row.student_no, name: row.name, section_id }),
				})
					.then<RowResult>(() => ({ ...row, status: 'success' as const }))
					.catch<RowResult>((err) => {
						const detail =
							err instanceof ApiError
								? (err.detail ?? `${err.status} ${err.statusText}`)
								: String(err);
						return { ...row, status: 'error' as const, detail };
					})
			)
		);

		results = [...parsed.filter((r) => r.status === 'invalid'), ...settled].sort(
			(a, b) => a.lineNo - b.lineNo
		);

		const succeeded = results.filter((r) => r.status === 'success').length;
		const failed = results.length - succeeded;
		isLoading = false;

		if (failed === 0) {
			toast.success(`Added ${succeeded} student${succeeded === 1 ? '' : 's'}`);
			await invalidateAll();
			isBulkDialogOpen = false;
			formText = '';
			results = null;
		} else {
			toast.error(`${succeeded} added, ${failed} failed`);
			if (succeeded > 0) await invalidateAll();
		}
	}
</script>

<Dialog.Content class="max-w-lg">
	<Dialog.Header>
		<Dialog.Title>Add students in bulk</Dialog.Title>
		<Dialog.Description>
			One student per line: <code>student_number,Full Name</code>
		</Dialog.Description>
	</Dialog.Header>

	<Label>Students</Label>
	<Textarea
		rows={10}
		bind:value={formText}
		class="font-mono text-sm"
		placeholder={"200011111,John Doe\n200011112,Jane Doe"}
		disabled={isLoading}
	/>

	{#if results !== null}
		{@const succeeded = results.filter((r) => r.status === 'success').length}
		{@const failed = results.length - succeeded}
		<div class="mt-1 space-y-1 text-sm">
			<p class="font-medium">
				{succeeded} added, {failed} failed
			</p>
			{#each results.filter((r) => r.status !== 'success') as row}
				<div class="flex items-start gap-2 rounded-md bg-destructive/10 px-2 py-1 text-destructive">
					{#if row.status === 'invalid'}
						<MdiAlertCircle class="mt-0.5 size-4 shrink-0" />
					{:else}
						<MdiClose class="mt-0.5 size-4 shrink-0" />
					{/if}
					<span>
						Line {row.lineNo}
						{#if row.student_no}(<code>{row.student_no}</code>){/if}
						— {row.detail}
					</span>
				</div>
			{/each}
		</div>
	{/if}

	<Dialog.Footer>
		<Button
			variant="outline"
			onclick={handleBulkAdd}
			disabled={isLoading || formText.trim() === ''}
		>
			{#if isLoading}
				Adding...
			{:else}
				Add students
			{/if}
		</Button>
		{#if results !== null && results.some((r) => r.status === 'success')}
			<div class="flex items-center gap-1 text-sm text-muted-foreground">
				<MdiCheck class="size-4 text-green-600" />
				{results.filter((r) => r.status === 'success').length} added successfully
			</div>
		{/if}
	</Dialog.Footer>
</Dialog.Content>
