<script lang="ts">
  let { supposedScans, onAccept, onReject, isCommitmentOngoing} = $props();
  import IconLabel from "~icons/mdi/label";

	import * as Dialog from "$lib/components/ui/dialog/index.ts";
	import { Input } from "$lib/components/ui/input/index.ts";
</script>


<Dialog.Content>
  <div class="flex flex-row w-full gap-x-1">
    <button class="button-secondary flex-1 text-sm"
            disabled={isCommitmentOngoing}
            onclick={onAccept}>
      {isCommitmentOngoing ? "Evaluating..." : "Accept and evaluate"}
    </button>
    <button class="button-destructive text-sm"
            onclick={onReject}>
      Discard
    </button>
  </div>
  
  {#each supposedScans as supposedScan}
    <div class="card relative
                flex flex-row justify-center items-start gap-x-1">
      {#if supposedScan.editing}
        <div class="absolute top-1 left-1 bg-white/80 flex flex-row items-center gap-1">
          <Input class="border px-1 text-sm w-12"
                type="text"
                bind:value={supposedScan.item_number}
                />
          <button class="button-outline text-xs"
                  onclick={() => supposedScan.editing = false}>
            Save
          </button>
        </div>
      
      {:else}
        <button class="absolute top-1 left-1
                      flex flex-row items-center ring-1 ring-border px-2 cursor-pointer rounded-sm
                      bg-white/80 *:opacity-80"
                  onclick={() => supposedScan.editing = true}>
          <IconLabel />
          <h4>{supposedScan.item_number}</h4>
        </button>
      {/if}
      
      <img class="block sm:max-w-3/4 md:max-w-2/3"
            src={`${supposedScan.image_directory}`}
            alt={supposedScan.item_number}
            />
    </div>
  {/each}
</Dialog.Content>