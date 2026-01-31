<script lang="ts">
	const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;

	import type { TestItem } from '$lib/types/types.ts';
	import * as Dialog from '$lib/components/ui/dialog/index.js';
	import { Label } from '$lib/components/ui/label/index.js';
	import { Input } from '$lib/components/ui/input/index.js';

	let formItemLabel: string = "";
	let formItemQuestion: string = "";
	let formItemIsProblemSolving: boolean = false;
	let formItemEARQ: string = "";

	async function addTestItem() {
		// TODO: finish
		try {
		} catch (e) {
		}
	}
</script>

<Dialog.Content>
	<Dialog.Header>
		<Dialog.Title>Add new test item</Dialog.Title>
	</Dialog.Header>
	<!-- TODO: finish -->
	<Dialog.Footer>
		<button class="button-primary" onclick={() => addTestItem()}>
			Add item
		</button>
	</Dialog.Footer>
</Dialog.Content>