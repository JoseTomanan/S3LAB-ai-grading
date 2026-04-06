<script lang="ts">
  import { API_URL } from '$lib/constants.ts';
  import { invalidateAll } from '$app/navigation';
  import { api, ApiError } from '$lib/utils/api.ts';
  import toast from 'svelte-5-french-toast';
  import * as Dialog from "$lib/components/ui/dialog/index.ts";
  import { Button } from "$lib/components/CustomButton/index.ts";
  import { Input } from "$lib/components/ui/input/index.ts";
  import { Label } from "$lib/components/ui/label/index.ts";

  let isOperationOngoing: boolean = $state(false);
  let formSectionName: string = $state("");

  async function addSection() {
    if (!formSectionName.trim()) {
      toast("Please enter a section name.", { icon: "⚠️" });
      return;
    }

    isOperationOngoing = true;
    try {
      await api(`${API_URL}/api/sections/`, {
        method: "POST",
        body: JSON.stringify({ section: formSectionName.trim() }),
      });
      toast.success("Section added successfully!");
      await invalidateAll();
    } catch (e) {
      toast.error(e instanceof ApiError
        ? (e.detail ? e.detail : `${e.status} ${e.statusText}`)
        : "Failed to add new section:\n" + String(e)
        );
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
  <Input id="section_name"
          type="text"
          placeholder="e.g., 3-Einstein"
          pattern="[a-zA-Z0-9\-]*"
          required
          bind:value={formSectionName}
        />

  <Dialog.Footer>
    <Dialog.Description>
      Please do not include spaces or underscores in the section name; instead, use pascal case (e.g., 3-JoseRizal).
    </Dialog.Description>
    <Button variant="outline" disabled={isOperationOngoing} onclick={addSection}>
      {isOperationOngoing ? "Creating..." : "Create"}
    </Button>
  </Dialog.Footer>
</Dialog.Content>
