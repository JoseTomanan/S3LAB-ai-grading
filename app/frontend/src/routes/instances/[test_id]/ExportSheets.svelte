<script lang="ts">
  import { API_URL } from '$lib/constants.ts';

  const { test_id } = $props();

  import { Button } from '$lib/components/CustomButton/index.ts';
  import * as Dialog from '$lib/components/ui/dialog/index.ts';
  import toast from 'svelte-5-french-toast';

  let isNotClicked: boolean = $state(true);
  let isLoading: boolean = $state(false);

  async function exportSheets() {
    isNotClicked = false;
    isLoading = true;

    try {
      const response = await fetch(
            `${API_URL}/api/test_instances/${test_id}/export`,
            {
              method: "GET",
              headers: { 'Accept': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' }
            }
            );
      
      switch (response.status) {
        case 200:
          const blob = await response.blob();
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          
          a.href = url;
          a.download = `${test_id}.xlsx`;
          document.body.appendChild(a);
          a.click();
          
          window.URL.revokeObjectURL(url);
          document.body.removeChild(a);
          break;
        
        default:
          toast.error(`${response.status} ${response.statusText}`);
      }
    } catch (e) {
      toast.error("Failed to export spreadsheet:\n"+e);
      isNotClicked = true;
    } finally {
      isLoading = false;
    }
  }
</script>


<Dialog.Content>
  {#if isNotClicked && !isLoading}
    <Dialog.Description>Click below to export spreadsheet.</Dialog.Description>
    <Button variant="outline"
            onclick={() => {exportSheets()}}
            disabled={isLoading}>
      {isLoading
        ? "Downloading..."
        : "Download spreadsheet"}
    </Button>
  {:else}
    <h4><b>{test_id}.xlsx</b> has been downloaded.</h4>
    <h6>NOTE: Items with no uploaded image/with an uploaded image but no evaluation are not scored.</h6>
    <h6>You may close this dialog.</h6>
  {/if}
</Dialog.Content>
