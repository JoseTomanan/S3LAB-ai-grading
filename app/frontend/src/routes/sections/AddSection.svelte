<script lang="ts">
  import { invalidateAll } from '$app/navigation';
  import { api, ApiError } from '$lib/utils/api.ts';
  import * as Dialog from "$lib/components/ui/dialog/index.ts";
  import { Button } from "$lib/components/ui/button/index.ts";
  import { Input } from "$lib/components/ui/input/index.ts";
  import { Label } from "$lib/components/ui/label/index.ts";

  let isOperationOngoing: boolean = $state(false);
  let formSectionName: string = $state("");

  async function addSection() {
    if (!formSectionName.trim()) {
      alert("Please enter a section name.");
      return;
    }

    isOperationOngoing = true;
    try {
      await api('/api/sections/', {
        method: "POST",
        body: JSON.stringify({ section: formSectionName.trim() }),
      });
      alert("Section added successfully!");
      await invalidateAll();
    } catch (e) {
      alert(e instanceof ApiError ? `${e.status} ${e.statusText}` : "Failed to add new section:\n" + e);
    } finally {
      isOperationOngoing = false;
    }
  }
</script>

<Dialog.Content>
  <Dialog.Header>
    <Dialog.Title>Add new section</Dialog.Title>
  </Dialog.Header>

  <Label for="section_name">Section name</Label>
  <Input
    id="section_name"
    type="text"
    placeholder="e.g. Grade 10 - Einstein"
    required
    bind:value={formSectionName}
  />

  <Dialog.Footer>
    <Button variant="outline" disabled={isOperationOngoing} onclick={addSection}>
      {isOperationOngoing ? "Adding..." : "Add section"}
    </Button>
  </Dialog.Footer>
</Dialog.Content>
