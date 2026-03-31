<script lang="ts">
	import { API_URL } from "$lib/constants.ts";
  let { supposedScans, onAccept, onReject, isCommitmentOngoing} = $props();

  import { getContext } from "svelte";
  import type { TestItemsContext } from '$lib/index.ts';
  import IconLabel from "~icons/mdi/label";
	import IconDiscard from "~icons/mdi/close";
	import * as Dialog from "$lib/components/ui/dialog/index.ts";
	import Card from "$lib/components/Card.svelte";

  let testItemsContext: TestItemsContext = getContext("testItemsContext");
  let validLabels = $derived(new Set(testItemsContext.items.map(i => i.label)));

  let activeScans = $derived(supposedScans.filter((s: { isDiscarded: boolean }) => !s.isDiscarded));

  let hasDuplicateLabels = $derived(
    new Set(activeScans.map((s: { item_number: string }) => s.item_number)).size !== activeScans.length
  );
  let isAcceptableLabels = $derived(
    activeScans.every((s: { item_number: string }) => validLabels.has(s.item_number))
  );

  const isButtonDisabled = $derived(isCommitmentOngoing || hasDuplicateLabels || !isAcceptableLabels || activeScans.length === 0);
</script>


<Dialog.Content showCloseButton={false}
                onEscapeKeydown={(e) => e.preventDefault()}>
  <div class="flex flex-row w-full gap-x-1">
    <button class="button-secondary flex-1 text-sm {isButtonDisabled ? "opacity-50" : ""}"
            disabled={isButtonDisabled}
            onclick={onAccept}>
      {#if isCommitmentOngoing}
        Identifying labels...
      {:else if activeScans.length === 0}
        <i>No boxes to evaluate</i>
      {:else if hasDuplicateLabels}
        <i>Please fix duplicate labels first</i>
      {:else if !isAcceptableLabels}
        <i>
          Labels must match test items:
          {Array.from(validLabels).join(", ")}
        </i>
      {:else}
        Accept and evaluate
      {/if}
    </button>
    <button class="button-destructive text-sm"
            onclick={onReject}>
      Discard
    </button>
  </div>
  
  <div class="max-h-[80vh] overflow-y-auto">
  {#each supposedScans as supposedScan}
    <Card class="relative flex flex-row justify-center items-start gap-x-1
                  {supposedScan.isDiscarded ? "opacity-40" : ""}">
      <div class="absolute top-1 left-1 bg-white/80 shadow-sm
                  flex flex-row items-center gap-x-0">
        <IconLabel class={validLabels.has(supposedScan.item_number) ? "" : "text-destructive"}/>
        <input class="w-8 px-1 py-0 leading-none bg-transparent border-none outline-none
                      [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
                type="text"
                bind:value={supposedScan.item_number}
                disabled={supposedScan.isDiscarded}
                />
      </div>

      <button class="absolute top-1 right-1 bg-white/80 shadow-sm p-0.5
                     {supposedScan.isDiscarded ? "text-destructive" : "opacity-60 hover:opacity-100"}"
              onclick={() => supposedScan.isDiscarded = !supposedScan.isDiscarded}
              title={supposedScan.isDiscarded ? "Restore box" : "Discard box"}>
        <IconDiscard class="size-4" />
      </button>

      <img class="block sm:max-w-3/4 md:max-w-2/3
                            {supposedScan.isDiscarded ? "grayscale" : ""}"
            src="{API_URL}{supposedScan.image_directory}"
            alt={supposedScan.item_number}
            />
    </Card>
  {/each}
  </div>
</Dialog.Content>