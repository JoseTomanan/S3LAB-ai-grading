<script lang="ts">
  import { onMount } from 'svelte';
  import { invalidateAll } from '$app/navigation';
  import { api, ApiError } from '$lib/utils/api.ts';
  import toast from 'svelte-5-french-toast';

  let { section_id = $bindable(), student_no, name = $bindable() } = $props();

  import * as Dialog from "$lib/components/ui/dialog/index.ts";
  import * as Select from "$lib/components/ui/select/index.ts";
  import Button from '$lib/components/ui/button/button.svelte';
  import { Input } from '$lib/components/ui/input/index.ts';
  import { Label } from '$lib/components/ui/label/index.ts';
	import { Spinner } from '$lib/components/ui/spinner/index.ts';

  import type { Section } from '$lib/index.ts';
	import SafeDelete from '$lib/components/SafeDelete.svelte';

  let isSectionsLoading = $state(false);
  let dropdownSections: Section[] = $state([]);

  let isRequestLoading = $state(false);

  let formName: string = $state(name.slice());
  let formSectionId: string = $state(section_id.toString())

  let isWantsToDelete: boolean = $state(false);
  $effect(() => {
    formName;
    formSectionId;
    isWantsToDelete = false;
  });

  const triggerContent = $derived(
        dropdownSections.find((f) => f.section_id.toString() == formSectionId)?.section_name ?? "Section..."
        );

  onMount(async () => {
    isSectionsLoading = true;
    try {
      dropdownSections = await api<Section[]>('/api/sections/');
    } catch (e) {
      toast.error(e instanceof ApiError ? `${e.status} ${e.statusText}` : "Failed to fetch sections.\n" + e);
    } finally {
      isSectionsLoading = false;
    }
  });

  async function editStudent() {
    if (formName == name && formSectionId == section_id.toString()) {
      toast("No changes were made.", { icon: "⚠️" });
      return;
    }
    isRequestLoading = true;
    try {
      await api(`/api/students/${student_no}`, {
        method: "PATCH",
        body: JSON.stringify({
          name: formName,
          section_id: parseInt(formSectionId),
        })
      });
      toast.success("Student updated successfully!");
      name = formName;
      section_id = Number(formSectionId);
      await invalidateAll();
    } catch (e) {
      toast.error(e instanceof ApiError ? `${e.status} ${e.statusText}` : "Failed to edit student: " + e);
    } finally {
      isRequestLoading = false;
    }
  }

  async function deleteStudent() {
    isRequestLoading = true;
    try {
      await api(`/api/students/${student_no}`, { method: "DELETE" });
      toast.success("Student deleted successfully!");
      section_id = -1;
      await invalidateAll();
    } catch (e) {
      toast.error(e instanceof ApiError ? `${e.status} ${e.statusText}` : "Failed to delete student: " + e);
    } finally {
      isRequestLoading = false;
    }
  }
</script>


<Dialog.Content>
  <Dialog.Header>
    <Dialog.Title>Edit student</Dialog.Title>
  </Dialog.Header>
  <Label id="student_no">Student number</Label>
  <Input id="student_no"
          placeholder="Student number..."
          value={student_no}
          disabled/>
  <Label id="name">Name</Label>
  <Input id="name"
          placeholder="Name..."
          bind:value={formName} />
  <Label id="section">Section</Label>
  <Select.Root type="single"
                name="section"
                bind:value={formSectionId}>
    <Select.Trigger class="w-full text-base md:text-sm" disabled={isSectionsLoading}>
      {isSectionsLoading ? "Loading..." : triggerContent}
    </Select.Trigger>
    <Select.Content>
      {#each dropdownSections as s (s.section_id)}
        <Select.Item value={s.section_id.toString()}
                      label={s.section_name}>
          {s.section_name} ({s.section_id})
        </Select.Item>
      {/each}
    </Select.Content>
  </Select.Root>
  <Dialog.Footer>
    <div class="flex flex-wrap gap-x-1 w-full">
      <Button variant="outline"
              class="flex-1"
              disabled={isRequestLoading}
              onclick={() => editStudent()}>
        {isRequestLoading ? "Saving..." : "Save changes"}
        {#if isRequestLoading}
          <Spinner />
        {/if}
      </Button>
      <SafeDelete toggle={isWantsToDelete}
                  onDelete={deleteStudent}
                  />
    </div>
  </Dialog.Footer>
</Dialog.Content>
