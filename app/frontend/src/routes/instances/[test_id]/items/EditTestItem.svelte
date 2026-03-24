<script lang="ts">
  import { API_URL } from '$lib/constants.ts';
  
  import { untrack } from 'svelte';
  import { invalidateAll } from '$app/navigation';
  import { api, ApiError } from '$lib/utils/api.ts';
  import toast from 'svelte-5-french-toast';
  import type { TestItem } from '$lib/index.ts';
  import * as Dialog from '$lib/components/ui/dialog/index.js';
  import { Button } from '$lib/components/ui/button/index.ts';
  import { Label } from '$lib/components/ui/label/index.js';
  import { Textarea } from '$lib/components/ui/textarea/index.js';
  import { Input } from '$lib/components/ui/input/index.ts';
	import SafeDelete from '$lib/components/SafeDelete.svelte';

  let { testItem, test_id } = $props<{
    testItem: TestItem,
    test_id: string
  }>();

  let formTestItem: TestItem = $state({
        item_id: testItem.item_id,
        label: testItem.label,
        question: testItem.question,
        is_problem_solving: testItem.is_problem_solving,
        expected_answer_rubric_questions: testItem.expected_answer_rubric_questions,
      });

  // NOTE: This resets the form on any testItem prop change, including invalidateAll().
  // Safe because the dialog is always closed before invalidateAll() runs.
  $effect(() => {
    const updated = {
      item_id: testItem.item_id,
      label: testItem.label,
      question: testItem.question,
      is_problem_solving: testItem.is_problem_solving,
      expected_answer_rubric_questions: testItem.expected_answer_rubric_questions,
    };
    untrack(() => { formTestItem = updated; });
  });

  let isWantsToDelete = $state(false);
  $effect(() => {
    formTestItem.label;
    formTestItem.question;
    formTestItem.expected_answer_rubric_questions;
    isWantsToDelete = false;
  });

  async function editTestItem(submittedTestItem: TestItem) {
    if (testItem == submittedTestItem) {
      toast("No changes were made.", { icon: "⚠️" });
      return;
    }

    try {
      const result = await api<{ item_id: string }>(`${API_URL}/api/test_items/${test_id}/items/${submittedTestItem.item_id}`, {
        method: "PATCH",
        body: JSON.stringify({
          label: submittedTestItem.label,
          question: submittedTestItem.question,
          expected_answer_rubric_questions: submittedTestItem.expected_answer_rubric_questions,
        }),
      });
      toast.success("Test item updated successfully!");
      await invalidateAll();
    } catch (e) {
      toast.error(e instanceof ApiError ? `${e.status} ${e.statusText}` : "Failed to edit test item:\n" + e);
    }
  }

  async function deleteTestItem() {
    try {
      await api(`${API_URL}/api/test_items/${test_id}/items/${formTestItem.item_id}`, { method: "DELETE" });
      toast.success("Test item deleted.");
      await invalidateAll();
    } catch (e) {
      toast.error(e instanceof ApiError ? `${e.status} ${e.statusText}` : "Failed to delete test item:\n" + e);
    }
  }
</script>


<Dialog.Content>
  <Dialog.Header>
    <Dialog.Title>
      Edit test item {testItem.label}
    </Dialog.Title>
    <Dialog.Description>
      For expected answer/rubric questions, add points in brackets at end. Example: "Correct setup [1pts]"
    </Dialog.Description>
  </Dialog.Header>
  <Label for="label">Label</Label>
  <Input id="label"
          bind:value={ formTestItem.label }
          required />
  <Label for="question">Question</Label>
  <Textarea id="question"
          rows={4}
          bind:value={ formTestItem.question }
          required />
  {#if testItem.is_problem_solving}
    <Label for="r_q">Rubric questions (separate with `; `)</Label>
    <Textarea id="r_q"
              rows={6}
              bind:value={ formTestItem.expected_answer_rubric_questions }
              required />
  {:else}
    <Label for="e_a">Expected answer</Label>
    <Textarea id="e_a"
              rows={4}
              bind:value={ formTestItem.expected_answer_rubric_questions }
              required />
  {/if}
  <Dialog.Footer>
    <div class="flex flex-wrap w-full gap-1.5">
      <Button variant="outline"
              class="flex-1"
              onclick={() => editTestItem(formTestItem)}>
        Save changes
      </Button>
      <SafeDelete toggle={isWantsToDelete}
                  onDelete={deleteTestItem}
                  />
    </div>
  </Dialog.Footer>
</Dialog.Content>