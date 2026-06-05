<script lang="ts">
  import { API_URL } from '$lib/constants.ts';
  import { invalidateAll } from '$app/navigation';
  import { api, ApiError } from '$lib/utils/api.ts';
  import toast from 'svelte-5-french-toast';
  import { Button } from '$lib/components/CustomButton/index.ts';
  import { Input } from '$lib/components/ui/input/index.ts';
  import { Label } from '$lib/components/ui/label/index.ts';

  interface Props { close: () => void }
  let { close }: Props = $props();

  let isOperationOngoing = $state(false);
  let formSectionName = $state('');

  async function addSection() {
    if (!formSectionName.trim()) {
      toast('Please enter a section name.', { icon: '⚠️' });
      return;
    }
    isOperationOngoing = true;
    try {
      await api(`${API_URL}/api/sections/`, {
        method: 'POST',
        body: JSON.stringify({ section: formSectionName.trim() }),
      });
      toast.success('Section added.');
      await invalidateAll();
      close();
    } catch (e) {
      toast.error(
        e instanceof ApiError
          ? (e.detail ? e.detail : `${e.status} ${e.statusText}`)
          : 'Failed to add section:\n' + String(e)
      );
    } finally {
      isOperationOngoing = false;
    }
  }
</script>

<div class="flex flex-col gap-4">
  <div>
    <Label for="section_name" class="text-[11px] font-semibold uppercase tracking-[0.04em] text-muted-foreground mb-1.5 block">
      Section name
    </Label>
    <Input
      id="section_name"
      type="text"
      placeholder="e.g. 3-JoseRizal"
      pattern="[a-zA-Z0-9\-]*"
      required
      bind:value={formSectionName}
    />
    <p class="text-[11px] text-muted-foreground mt-1.5">
      No spaces or underscores — use pascal case (e.g. 3-JoseRizal).
    </p>
  </div>

  <div class="flex gap-2 mt-1">
    <Button variant="ghost" class="flex-1" onclick={close}>Cancel</Button>
    <Button variant="primary" class="flex-1" disabled={isOperationOngoing} onclick={addSection}>
      {isOperationOngoing ? 'Creating…' : 'Save'}
    </Button>
  </div>
</div>
