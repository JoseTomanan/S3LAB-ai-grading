<script lang="ts">
  import { API_URL } from '$lib/constants.ts';
  import { onMount } from "svelte";
  import { invalidateAll } from '$app/navigation';
  import { api, ApiError } from '$lib/utils/api.ts';
  import toast from 'svelte-5-french-toast';
  import type { Section } from "$lib/index.ts";

  import * as Dialog from "$lib/components/ui/dialog/index.js";
  import * as Select from "$lib/components/ui/select/index.ts";
  import { Label } from '$lib/components/ui/label/index.js';
  import { Input } from '$lib/components/ui/input/index.js';
  import { Button } from '$lib/components/CustomButton/index.ts';

  let isOperationOngoing = $state(false);
  
  let dropdownItems: Section[] = $state([]);

  let newInstanceName: string = $state("");
  let newInstanceSectionId: string = $state("");

  const triggerContent = $derived(
        dropdownItems.find((f) => f.section_id.toString() == newInstanceSectionId)?.section_name ?? "Section..."
        );

  onMount(async () => {
    try {
      dropdownItems = await api<Section[]>(`${API_URL}/api/sections/`);
    } catch (e) {
      toast.error(e instanceof ApiError
        ? (e.detail ? e.detail : `${e.status} ${e.statusText}`)
        : "Failed to fetch sections.\n" + String(e));
    }
  });

  async function addNewTestInstance(name: string, section_id: string) {
    if (newInstanceName === "" || newInstanceSectionId === "") {
      toast("Please do not leave any fields empty");
      return;
    }

    isOperationOngoing = true;
    try {
      await api(`${API_URL}/api/test_instances`, {
        method: "POST",
        body: JSON.stringify({
          name,
          section_id,
          date: new Date().toISOString(),
        }),
      });
      toast.success("Addition successful.");
      await invalidateAll();
    } catch (e) {
      toast.error(e instanceof ApiError
        ? (e.detail ? e.detail : `${e.status} ${e.statusText}`)
        : "Failed to add new test instance. Check your network connection and try again.");
    } finally {
      isOperationOngoing = false;
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
          pattern="[a-zA-Z0-9\-]*"
          required
          bind:value={newInstanceName}
        />
  
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
    <Dialog.Description>
      Please do not include spaces or underscores in the section name; instead, use hyphens.
      Note that the name and section cannot be changed after creation.
    </Dialog.Description>
    <Button variant="outline"
            disabled={isOperationOngoing}
            onclick={() => addNewTestInstance(newInstanceName, newInstanceSectionId)}>
      {isOperationOngoing ? "Creating..." : "Create"}
    </Button>
  </Dialog.Footer>
</Dialog.Content>