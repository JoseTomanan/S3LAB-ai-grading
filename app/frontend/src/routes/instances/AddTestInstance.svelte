<script lang="ts">
  import { onMount } from "svelte";
  import { invalidateAll } from '$app/navigation';
  import { api, ApiError } from '$lib/utils/api.ts';
  import type { Section } from "$lib/index.ts";

  import * as Dialog from "$lib/components/ui/dialog/index.js";
  import * as Select from "$lib/components/ui/select/index.ts";
  import { Label } from '$lib/components/ui/label/index.js';
  import { Input } from '$lib/components/ui/input/index.js';
  import { Button } from '$lib/components/ui/button/index.ts';

  let isItemsLoading: boolean = $state(true);
  let dropdownItems: Section[] = $state([]);

  // svelte-ignore non_reactive_update
  let newInstanceName: string = "";
  let newInstanceSectionId: string = $state("");

  const triggerContent = $derived(
        dropdownItems.find((f) => f.section_id.toString() == newInstanceSectionId)?.section_name ?? "Section..."
        );

  onMount(async () => {
    isItemsLoading = true;
    try {
      dropdownItems = await api<Section[]>('/api/sections/');
    } catch (e) {
      alert(e instanceof ApiError ? `${e.status} ${e.statusText}` : "Failed to fetch sections.\n" + e);
    } finally {
      isItemsLoading = false;
    }
  });

  async function addNewTestInstance(name: string, section_id: string) {
    try {
      await api('/api/test_instances', {
        method: "POST",
        body: JSON.stringify({
          name,
          section_id,
          date: new Date().toISOString(),
        }),
      });
      alert("Addition successful.");
      await invalidateAll();
    } catch (e) {
      alert(e instanceof ApiError
        ? `Addition fail: ${e.status} ${e.statusText}`
        : "Failed to add new test instance. Check your network connection and try again.");
    }
  }
</script>


<Dialog.Content>
  <Dialog.Header>
    <Dialog.Title>Add new test instance</Dialog.Title>
  </Dialog.Header>
  <Label for="name">Test name</Label>
  <Input id="name" type="text"
          placeholder="Test name..."
          required
          bind:value={newInstanceName}/>
  <Label for="section">Section</Label>
  <Select.Root type="single"
                name="section"
                bind:value={newInstanceSectionId}>
    <Select.Trigger class="w-full text-base md:text-sm">
      {triggerContent}
    </Select.Trigger>
    <Select.Content>
      {#each dropdownItems as item (item.section_id)}
        <Select.Item value={item.section_id.toString()}
                      label={item.section_name}>
          {item.section_name} ({item.section_id})
        </Select.Item>
      {/each}
    </Select.Content>
  </Select.Root>
  <Dialog.Footer>
    <Button variant="outline"
            onclick={() => addNewTestInstance(newInstanceName, newInstanceSectionId)}>
      Save changes
  </Button>
    <Dialog.Description>Note that the name and section cannot be changed after creation.</Dialog.Description>
  </Dialog.Footer>
</Dialog.Content>