<script lang="ts">
  import { API_URL } from '$lib/constants.ts';
  import * as Dialog from '$lib/components/ui/dialog/index.ts';
  import toast from 'svelte-5-french-toast';
  import IconTable from '~icons/mdi/table-export';

  const { test_id } = $props();

  let isNotClicked = $state(true);
  let isLoading = $state(false);

  async function exportSheets() {
    isNotClicked = false;
    isLoading = true;
    try {
      const response = await fetch(`${API_URL}/api/test_instances/${test_id}/export`, {
        method: 'GET',
        headers: { Accept: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' },
      });
      if (response.status === 200) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${test_id}.xlsx`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      } else {
        toast.error(`${response.status} ${response.statusText}`);
      }
    } catch (e) {
      toast.error('Failed to export spreadsheet:\n' + e);
      isNotClicked = true;
    } finally {
      isLoading = false;
    }
  }
</script>

<Dialog.Root onOpenChange={() => { isNotClicked = true; }}>
  <Dialog.Trigger
    class="size-9 rounded-full bg-background border border-border flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-border transition-colors"
    title="Export spreadsheet"
  >
    <IconTable class="size-[18px]" />
  </Dialog.Trigger>

  <Dialog.Content>
    <Dialog.Header>
      <Dialog.Title>Export spreadsheet</Dialog.Title>
    </Dialog.Header>
    {#if isNotClicked && !isLoading}
      <Dialog.Description>Download scores as an Excel spreadsheet.</Dialog.Description>
      <button
        class="w-full py-2.5 px-4 rounded-md bg-foreground text-background text-sm font-semibold hover:bg-foreground/80 transition-colors disabled:opacity-50"
        onclick={exportSheets}
        disabled={isLoading}
      >
        Download spreadsheet
      </button>
    {:else}
      <p class="text-sm font-medium"><b>{test_id}.xlsx</b> has been downloaded.</p>
      <p class="text-xs text-muted-foreground">Items with no uploaded image or no evaluation are not scored. You may close this dialog.</p>
    {/if}
  </Dialog.Content>
</Dialog.Root>
