<script lang="ts">
  import { API_URL } from '$lib/constants.ts';
  import { onMount } from 'svelte';
  import { invalidateAll } from '$app/navigation';
  import { api, ApiError } from '$lib/utils/api.ts';
  import toast from 'svelte-5-french-toast';
  import type { Section } from '$lib/index.ts';

  import * as Select from '$lib/components/ui/select/index.ts';
  import { Label } from '$lib/components/ui/label/index.js';
  import { Input } from '$lib/components/ui/input/index.js';
  import { Button } from '$lib/components/CustomButton/index.ts';

  interface Props { close: () => void }
  let { close }: Props = $props();

  let isOperationOngoing = $state(false);
  let dropdownItems: Section[] = $state([]);
  let newInstanceName = $state('');
  let newInstanceSectionId = $state('');

  const triggerContent = $derived(
    dropdownItems.find((f) => f.section_id.toString() === newInstanceSectionId)?.section_name ??
      'Select section…'
  );

  onMount(async () => {
    try {
      dropdownItems = await api<Section[]>(`${API_URL}/api/sections/`);
    } catch (e) {
      toast.error(
        e instanceof ApiError
          ? (e.detail ? e.detail : `${e.status} ${e.statusText}`)
          : 'Failed to fetch sections.\n' + String(e)
      );
    }
  });

  async function submit() {
    if (!newInstanceName || !newInstanceSectionId) {
      toast('Please fill in all fields');
      return;
    }
    isOperationOngoing = true;
    try {
      await api(`${API_URL}/api/test_instances`, {
        method: 'POST',
        body: JSON.stringify({
          name: newInstanceName,
          section_id: newInstanceSectionId,
          date: new Date().toISOString(),
        }),
      });
      toast.success('Test instance created.');
      await invalidateAll();
      close();
    } catch (e) {
      toast.error(
        e instanceof ApiError
          ? (e.detail ? e.detail : `${e.status} ${e.statusText}`)
          : 'Failed to add test instance.'
      );
    } finally {
      isOperationOngoing = false;
    }
  }
</script>

<div class="flex flex-col gap-4">
  <div>
    <Label for="inst-name" class="text-[11px] font-semibold uppercase tracking-[0.04em] text-muted-foreground mb-1.5 block">
      Name
    </Label>
    <Input
      id="inst-name"
      type="text"
      placeholder="e.g. Algebra Quiz 1"
      bind:value={newInstanceName}
    />
  </div>

  <div>
    <Label for="inst-section" class="text-[11px] font-semibold uppercase tracking-[0.04em] text-muted-foreground mb-1.5 block">
      Section
    </Label>
    <Select.Root type="single" name="inst-section" bind:value={newInstanceSectionId}>
      <Select.Trigger class="w-full">{triggerContent}</Select.Trigger>
      <Select.Content>
        {#each dropdownItems as item (item.section_id)}
          <Select.Item value={item.section_id.toString()} label={item.section_name}>
            {item.section_name}
          </Select.Item>
        {/each}
      </Select.Content>
    </Select.Root>
    <p class="text-[11px] text-muted-foreground mt-1.5">
      Name and section cannot be changed after creation.
    </p>
  </div>

  <div class="flex gap-2 mt-1">
    <Button variant="ghost" class="flex-1" onclick={close}>Cancel</Button>
    <Button variant="primary" class="flex-1" disabled={isOperationOngoing} onclick={submit}>
      {isOperationOngoing ? 'Creating…' : 'Save'}
    </Button>
  </div>
</div>
