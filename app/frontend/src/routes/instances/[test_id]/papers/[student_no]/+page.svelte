<script lang="ts">
  const { data } = $props();

  import { API_BASE_URL } from '$lib/constants.ts';

  import MdiPaperOff from '~icons/mdi/paper-off';
  import MdiImagePlus from '~icons/mdi/image-plus';
  import MdiCrop from '~icons/mdi/crop';

  import type { StudentAnswer } from '$lib/index.ts';
  import { Label } from '$lib/components/ui/label/index.js';
	import SafeDelete from '$lib/components/SafeDelete.svelte';

  if (!data.student_items)
    throw new Error("Student items failed to load");

  let studentItems: (StudentAnswer & {label: string})[] = $state(data.student_items);
  console.log(studentItems);

  let isWantsToDelete: boolean = $state(false);

  async function deleteAnswer(item_id: number) {
    console.log(item_id);
    const response = await fetch(`${API_BASE_URL}/api/student_answers/${item_id}/${data.student_no}`, { method: "DELETE" });
    switch (response.status) {
      case 204:
        alert(`Deletion of ${item_id} for ${data.student_no} successful.`);
        window.location.reload();
        break;
      default:
        alert(`${response.status} ${response.statusText}`);
    }
  }
</script>


<div class="flex flex-col gap-3">
  <span class="flex flex-row justify-between items-center">
    <span class="flex flex-wrap items-baseline gap-x-4 [&>h5]:opacity-60">
      <h1>Test answers</h1>
      <h5>{data.student_no}</h5>
    </span>
    <a class="button-secondary"
        href="/instances/{data.test_id}/papers/{data.student_no}/process">
      <MdiImagePlus class="size-5 mx-2"/>
    </a>
  </span>
  {#if studentItems.length == 0}
    <p>Nothing to see here. <br>If this is a mistake, check your network connection.</p>
  {:else}
    <div class="overflow-y-auto space-y-2">
    {#each studentItems as studentItem}
      <div class="card space-y-1">
        <Label for={studentItem.label} class="flex flex-row justify-between">
          {studentItem.label}
          <span class="flex flex-row space-x-1.5">
            <SafeDelete toggle={isWantsToDelete}
                        onDelete={() => deleteAnswer(studentItem.item_id)}
                        size={4}
                        />
            <a href="/instances/{data.test_id}/papers/{data.student_no}/manual?item_id={studentItem.item_id}"
                  class="button-outline">
              <MdiCrop/>
            </a>
          </span>
        </Label>
        <div class="flex justify-center items-center">
          {#if studentItem.image_directory == ""}
            <MdiPaperOff class="size-8 opacity-50" />
          {:else}
            <img class="size-fill md:max-w-1/2"
                  src={`${API_BASE_URL}${studentItem.image_directory}`}
                  alt={studentItem.label}/>
          {/if}
        </div>
      </div>
    {/each}
    </div>
  {/if}
</div>
