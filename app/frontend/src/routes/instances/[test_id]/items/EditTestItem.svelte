<script lang="ts">
    
  import MdiDelete from "~icons/mdi/delete";

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

  let isWantsToDelete = $state(false);
  $effect(() => {
    formTestItem.label;
    formTestItem.question;
    formTestItem.expected_answer_rubric_questions;
    isWantsToDelete = false;
  });

  async function editTestItem(submittedTestItem: TestItem) {
    if (testItem == submittedTestItem) {
      alert("No changes were made.");
      return;
    }

    try {
      const formBody = {
            label: submittedTestItem.label,
            question: submittedTestItem.question,
            expected_answer_rubric_questions: submittedTestItem.expected_answer_rubric_questions,
            };

      console.log(formBody);
      
      const response = await fetch(
            `/api/test_items/${test_id}/items/${submittedTestItem.item_id}`,
            {
              method: "PATCH",
              headers: {'Content-Type': 'application/json',},
              body: JSON.stringify(formBody),
            }
            );
      
      switch (response.status) {
        case 200:
          const result = await response.json();
          alert("Success: " + result.item_id);
          window.location.reload();
          break;
        default:
          alert(`${response.status} ${response.statusText}`);
      }
    } catch (e) {
      alert("Failed to edit test item:\n"+e);
    }
  }

  async function deleteTestItem() {
    try {
      const response = await fetch(
            `/api/test_items/${test_id}/items/${formTestItem.item_id}`,
            {
              method: "DELETE",
              headers: {'Content-Type': 'application/json',},
            }
            );
      
      switch (response.status) {
        case 204:
          alert("Delete success.");
          window.location.reload();
          break;
        default:
          alert(`${response.status} ${response.statusText}`);
      }
    } catch (e) {
      alert("Failed to delete test item:\n"+e)
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