<script lang="ts">
	import { API_BASE_URL } from '$lib/constants.ts';

	const { test_id } = $props();
	
	import { Button } from '$lib/components/ui/button/index.ts';
	import * as Dialog from '$lib/components/ui/dialog/index.ts';

	let isNotClicked: boolean = $state(true);
	let isLoading: boolean = $state(false);

	async function exportSheets() {
		isNotClicked = false;
		isLoading = true;

		try {
			const response = await fetch(
						`${API_BASE_URL}/api/test_instances/${test_id}/export`,
						{
							method: "GET",
							headers: { 'Accept': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' }
						}
						);
			
			if (response.ok) {
				const blob = await response.blob();
				
				const url = window.URL.createObjectURL(blob);
				const a = document.createElement('a');
				
				a.href = url;
				a.download = 'report.xlsx';
				document.body.appendChild(a);
				a.click();
				
				window.URL.revokeObjectURL(url);
				document.body.removeChild(a);
			} else
				alert(`${response.status} ${response.statusText}`);
		} catch (e) {
			alert("There was an error in processing the request. Please try again.");
			isNotClicked = true;
		} finally {
			isLoading = false;
		}
	}
</script>


<Dialog.Content>
	{#if isNotClicked}
		<h4>Click below to export test:</h4>
		<Button variant="outline"
						onclick={() => {exportSheets()}}>
			Download spreadsheet
		</Button>
	{:else if isLoading}
		<p>Loading...</p>
	{:else}
		<h6>Spreadsheet download has been initialized.</h6>
		<p>You may close this dialog.</p>
	{/if}
</Dialog.Content>
