<script lang="ts">
  import { API_URL } from '$lib/constants.ts';
  import { invalidateAll } from '$app/navigation';
  import { api, ApiError } from '$lib/utils/api.ts';
  import toast from 'svelte-5-french-toast';

  import * as RadioGroup from '$lib/components/ui/radio-group/index.ts';
  import { Button } from '$lib/components/CustomButton/index.js';
  import { Label } from '$lib/components/ui/label/index.js';
  import { Input } from '$lib/components/ui/input/index.js';
  import { Textarea } from '$lib/components/ui/textarea/index.ts';

  interface Props { test_id: string; close: () => void }
  let { test_id, close }: Props = $props();

  let isOperationOngoing = $state(false);
  let formItemLabel = $state('');
  let formItemQuestion = $state('');
  let formItemIsProblemSolving = $state(false);
  let formItemRQ = $state([{ question: '', points: 1 }]);
  let formItemEA = $state({ question: '', points: 1 });

  $effect(() => {
    formItemIsProblemSolving;
    formItemRQ = [{ question: '', points: 1 }];
    formItemEA = { question: '', points: 1 };
  });

  async function addTestItem() {
    if (!formItemLabel || !formItemQuestion) {
      toast('Please fill in all fields');
      return;
    }
    isOperationOngoing = true;
    const earq = formItemIsProblemSolving
      ? formItemRQ
          .filter((item) => item.points !== 0 && item.question.trim() !== '')
          .map((item) => `${item.question} [${item.points}pts]`)
          .join(';')
      : `${formItemEA.question} [${formItemEA.points}pts]`;
    try {
      await api(`${API_URL}/api/test_items/${test_id}/items`, {
        method: 'POST',
        body: JSON.stringify({
          label: formItemLabel,
          question: formItemQuestion,
          is_problem_solving: formItemIsProblemSolving,
          expected_answer_rubric_questions: earq,
        }),
      });
      toast.success('Test item added.');
      await invalidateAll();
      close();
    } catch (e) {
      toast.error(
        e instanceof ApiError
          ? (e.detail ? e.detail : `${e.status} ${e.statusText}`)
          : 'Failed to add test item:\n' + e
      );
    } finally {
      isOperationOngoing = false;
    }
  }
</script>

<div class="flex flex-col gap-4 max-h-[60dvh] overflow-y-auto pr-1">
  <div>
    <Label for="item_label" class="text-[11px] font-semibold uppercase tracking-[0.04em] text-muted-foreground mb-1.5 block">Label</Label>
    <Input id="item_label" placeholder="e.g. 1 or A" bind:value={formItemLabel} />
  </div>

  <div>
    <Label for="item_question" class="text-[11px] font-semibold uppercase tracking-[0.04em] text-muted-foreground mb-1.5 block">Question</Label>
    <Textarea id="item_question" rows={3} placeholder="Question text…" bind:value={formItemQuestion} />
  </div>

  <div>
    <Label class="text-[11px] font-semibold uppercase tracking-[0.04em] text-muted-foreground mb-1.5 block">Type</Label>
    <RadioGroup.Root
      value="short_form"
      class="flex flex-row gap-4"
      onValueChange={(v) => { formItemIsProblemSolving = v === 'prob_sol'; }}
    >
      <span class="flex items-center gap-2">
        <RadioGroup.Item value="short_form" id="short_form" />
        <Label for="short_form" class="font-normal">Short form</Label>
      </span>
      <span class="flex items-center gap-2">
        <RadioGroup.Item value="prob_sol" id="prob_sol" />
        <Label for="prob_sol" class="font-normal">Problem solving</Label>
      </span>
    </RadioGroup.Root>
  </div>

  {#if formItemIsProblemSolving}
    <div>
      <div class="flex justify-between items-center mb-1.5">
        <Label class="text-[11px] font-semibold uppercase tracking-[0.04em] text-muted-foreground">Rubric questions</Label>
        <button
          class="text-[12px] font-semibold text-primary-700 hover:underline"
          onclick={() => { formItemRQ.push({ question: '', points: 1 }); }}
        >+ Add</button>
      </div>
      <div class="flex flex-col gap-1.5">
        {#each formItemRQ as item, idx}
          <div class="flex gap-1.5">
            <Input class="flex-1" placeholder="Rubric {idx + 1}…" pattern="[^\[\];]*" bind:value={item.question} />
            <Input class="w-16" type="number" placeholder="pts" bind:value={item.points} />
          </div>
        {/each}
      </div>
    </div>
  {:else}
    <div>
      <Label class="text-[11px] font-semibold uppercase tracking-[0.04em] text-muted-foreground mb-1.5 block">Expected answer &amp; points</Label>
      <div class="flex gap-1.5">
        <Input class="flex-1" placeholder="Expected answer…" pattern="[^\[\];]*" bind:value={formItemEA.question} />
        <Input class="w-16" type="number" bind:value={formItemEA.points} />
      </div>
    </div>
  {/if}
</div>

<div class="flex gap-2 mt-5">
  <Button variant="ghost" class="flex-1" onclick={close}>Cancel</Button>
  <Button variant="primary" class="flex-1" disabled={isOperationOngoing} onclick={addTestItem}>
    {isOperationOngoing ? 'Creating…' : 'Save'}
  </Button>
</div>
