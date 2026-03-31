<script lang="ts">
  import { API_URL } from '$lib/constants.ts';

  const { test_id, isAddDialogOpen } = $props();
  import { invalidateAll } from '$app/navigation';
  import { api, ApiError } from '$lib/utils/api.ts';
  import toast from 'svelte-5-french-toast';

  import * as Dialog from '$lib/components/ui/dialog/index.js';
  import * as RadioGroup from '$lib/components/ui/radio-group/index.ts';
  import { Button } from '$lib/components/ui/button/index.js';
  import { Label } from '$lib/components/ui/label/index.js';
  import { Input } from '$lib/components/ui/input/index.js';
  import { Textarea } from '$lib/components/ui/textarea/index.ts';

  let isOperationOngoing: boolean = $state(false);

  let formItemLabel: string = $state("");
  let formItemQuestion: string = $state("");
  let formItemIsProblemSolving: boolean = $state(false);
  let formItemRQ: {question: string, points: number}[] = $state([{question: "", points: 1}]);
  let formItemEA: {question: string, points: number} = $state({question: "", points: 1});

  let submittableEARQ: string = "";

  async function addTestItem() {
    if (formItemLabel === "" || formItemQuestion === "") {
      toast("Please do not leave any fields empty")
      return;
    }

    isOperationOngoing = true;

    if (formItemIsProblemSolving) {
      // Convert the rubric questions array for problem-solving items into a semicolon-delimited string,
      // including only questions with nonzero points and non-blank text, each formatted as "question [Npts]".
      submittableEARQ = formItemRQ
            .filter(item => item.points != 0 && item.question.trim() !== "")
            .map(item => `${item.question} [${item.points}pts]`)
            .join(';');
    } else {
      submittableEARQ = `${formItemEA.question} [${formItemEA.points}pts]`;
    }

    try {
      const result = await api<{ items: string }>(`${API_URL}/api/test_items/${test_id}/items`, {
        method: "POST",
        body: JSON.stringify({
          label: formItemLabel,
          question: formItemQuestion,
          is_problem_solving: formItemIsProblemSolving,
          expected_answer_rubric_questions: submittableEARQ,
        }),
      });
      toast.success("Test item added successfully!");
      await invalidateAll();
    } catch (e) {
      toast.error(e instanceof ApiError ? `${e.status} ${e.statusText}` : "Failed to add new test item:\n" + e);
    } finally {
      isOperationOngoing = false;
    }
  }

  $effect(() => {
    formItemIsProblemSolving;
    formItemRQ = [{question: "", points: 1}];
    formItemEA = {question: "", points: 1};
  })

  $effect(() => {
    if (isAddDialogOpen) {
      formItemLabel = "";
      formItemQuestion = "";
      formItemIsProblemSolving = false;
      formItemRQ = [{question: "", points: 1}];
      formItemEA = {question: "", points: 1};
    }
  });
</script>



<Dialog.Content class="max-h-[80dvh] overflow-scroll">
  <Dialog.Header>
    <Dialog.Title>Add new test item</Dialog.Title>
  </Dialog.Header>
  
  <Label for="item_label">Item label</Label>
  <Input id="item_label"
          placeholder="Label..."
          bind:value={formItemLabel}
          />
  
  <Label for="item_question">Question</Label>
  <Textarea id="item_question" rows={4}
            placeholder="Question..."
            bind:value={formItemQuestion}
          />
  
  <Label>Type of question</Label>
  <RadioGroup.Root value="short_form"
                  class="flex flex-row justify-between
                          *:flex *:flex-row *:gap-2
                          [&>*>Label]:font-normal"
                  onValueChange={(v) => formItemIsProblemSolving = (v == "prob_sol")}>
    <span class="w-2/5">
      <RadioGroup.Item value="short_form" id="short_form"/>
      <Label for="short_form">Short form</Label>
    </span>
    <span class="w-3/5">
      <RadioGroup.Item value="prob_sol" id="prob_sol"/>
      <Label for="prob_sol">Problem solving</Label>
    </span>
  </RadioGroup.Root>
  
  {#if formItemIsProblemSolving}
    <Label for="r_q"
            class="flex flex-row justify-between">
      <span>
        Rubric questions
        <button class="rounded-sm px-1 py-0 my-0 ml-0.5
                bg-secondary text-secondary-foreground/90"
                onclick={() => {formItemRQ.push({question: "", points: 1})}}>
          + Add
        </button>
      </span>
      <span>Points</span>
    </Label>
    <div class="space-y-1">
      {#each formItemRQ as item, idx}
        <span class="flex flex-row gap-1">
          <Input id="r_q"
                  class="w-5/6"
                  placeholder="Rubric {idx+1}..."
                  pattern="[^\[\];]*"
                  bind:value={item.question} />
          <Input id="r_q" class="w-1/6"
                  type="number"
                  placeholder="Points..."
                  bind:value={item.points} />
        </span>
      {/each}
    </div>
  
  {:else}
    <Label for="e_a"
            class="flex flex-row justify-between">
      <span>Expected answer</span>
      <span>Points</span>
    </Label>
    <span class="flex flex-row gap-1">
      <Input id="e_a"
              class="w-5/6"
              placeholder="Expected answer..."
              pattern="[^\[\];]*"
              bind:value={formItemEA.question} />
      <Input id="e_a"
              class="w-1/6"
              type="number"
              bind:value={formItemEA.points} />
    </span>
  {/if}
  
  <Dialog.Footer>
    <Button variant="outline"
            disabled={isOperationOngoing}
            onclick={() => addTestItem()}>
      {isOperationOngoing ? "Creating..." : "Create item"}
    </Button>
  </Dialog.Footer>
</Dialog.Content>