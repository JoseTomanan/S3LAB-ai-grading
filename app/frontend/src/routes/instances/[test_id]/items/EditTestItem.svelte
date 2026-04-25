<script lang="ts">
  import { API_URL } from '$lib/constants.ts';
  import { untrack } from 'svelte';
  import { invalidateAll } from '$app/navigation';
  import { api, ApiError } from '$lib/utils/api.ts';
  import toast from 'svelte-5-french-toast';
  import type { TestItem } from '$lib/index.ts';

  import { Button } from '$lib/components/CustomButton/index.ts';
  import { Label } from '$lib/components/ui/label/index.js';
  import { Textarea } from '$lib/components/ui/textarea/index.js';
  import { Input } from '$lib/components/ui/input/index.ts';
  import SafeDelete from '$lib/components/SafeDelete.svelte';

  interface Props { testItem: TestItem; test_id: string; close: () => void }
  let { testItem, test_id, close }: Props = $props();

  let isOperationOngoing = $state(false);
  let formTestItem = $state({ ...untrack(() => testItem) });

  $effect(() => {
    const updated = { ...testItem };
    untrack(() => { formTestItem = updated; });
  });

  let isWantsToDelete = $state(false);
  $effect(() => {
    formTestItem.label;
    formTestItem.question;
    formTestItem.expected_answer_rubric_questions;
    isWantsToDelete = false;
  });

  async function editTestItem() {
    if (testItem === formTestItem) { toast('No changes were made.', { icon: '⚠️' }); return; }
    if (!formTestItem.label || !formTestItem.question || !formTestItem.expected_answer_rubric_questions) {
      toast('Please fill in all fields.');
      return;
    }
    if (isOperationOngoing) return;
    isOperationOngoing = true;
    try {
      await api(`${API_URL}/api/test_items/${test_id}/items/${formTestItem.item_id}`, {
        method: 'PATCH',
        body: JSON.stringify({
          label: formTestItem.label,
          question: formTestItem.question,
          expected_answer_rubric_questions: formTestItem.expected_answer_rubric_questions,
        }),
      });
      toast.success('Test item updated.');
      await invalidateAll();
      close();
    } catch (e) {
      toast.error(
        e instanceof ApiError
          ? (e.detail ? e.detail : `${e.status} ${e.statusText}`)
          : 'Failed to edit test item:\n' + e
      );
    } finally {
      isOperationOngoing = false;
    }
  }

  async function deleteTestItem() {
    try {
      await api(`${API_URL}/api/test_items/${test_id}/items/${formTestItem.item_id}`, { method: 'DELETE' });
      toast.success('Test item deleted.');
      await invalidateAll();
      close();
    } catch (e) {
      toast.error(
        e instanceof ApiError
          ? (e.detail ? e.detail : `${e.status} ${e.statusText}`)
          : 'Failed to delete test item:\n' + e
      );
    }
  }
</script>

<div class="flex flex-col gap-4 max-h-[60dvh] overflow-y-auto pr-1">
  <div>
    <Label for="edit_label" class="text-[11px] font-semibold uppercase tracking-[0.04em] text-muted-foreground mb-1.5 block">Label</Label>
    <Input id="edit_label" bind:value={formTestItem.label} required />
  </div>

  <div>
    <Label for="edit_question" class="text-[11px] font-semibold uppercase tracking-[0.04em] text-muted-foreground mb-1.5 block">Question</Label>
    <Textarea id="edit_question" rows={4} bind:value={formTestItem.question} required />
  </div>

  <div>
    <Label for="edit_earq" class="text-[11px] font-semibold uppercase tracking-[0.04em] text-muted-foreground mb-1.5 block">
      {testItem.is_problem_solving ? 'Rubric questions (separate with "; ")' : 'Expected answer'}
    </Label>
    <p class="text-[11px] text-muted-foreground mb-1.5">Add points in brackets, e.g. "Correct setup [1pts]"</p>
    <Textarea id="edit_earq" rows={testItem.is_problem_solving ? 6 : 4} bind:value={formTestItem.expected_answer_rubric_questions} required />
  </div>
</div>

<div class="flex gap-2 mt-5">
  <SafeDelete onDelete={deleteTestItem} />
  <Button variant="primary" class="flex-1" disabled={isOperationOngoing} onclick={editTestItem}>
    {isOperationOngoing ? 'Saving…' : 'Save changes'}
  </Button>
</div>
