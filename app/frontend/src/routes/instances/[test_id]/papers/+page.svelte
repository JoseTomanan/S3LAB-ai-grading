<script lang="ts">
  import { API_URL } from '$lib/constants.ts';
  import { onDestroy, untrack } from 'svelte';
  import type { GetEvaluationsResponse } from '$lib/index.ts';
  import { Spinner } from '$lib/components/ui/spinner/index.ts';
  import { api } from '$lib/utils/api.ts';
  import { createPoller } from '$lib/utils/poller.ts';
  import BottomSheet from '$lib/components/BottomSheet.svelte';
  import AddNewStudent from '@/routes/sections/[section_id]/AddNewStudent.svelte';
  import IconPlus from '~icons/mdi/plus';
  import IconDone from '~icons/mdi/account-check-outline';
  import IconUndone from '~icons/mdi/account-outline';

  const { data } = $props();

  let showAddStudent = $state(false);
  let perStudentStatuses: GetEvaluationsResponse[] = $state(untrack(() => data.statuses));
  $effect(() => { perStudentStatuses = data.statuses; });

  const isPolling = $derived(
    perStudentStatuses.some((s) => s.has_any_answer && !s.is_all_answers_evaluated)
  );

  const poller = createPoller(async () => {
    const result = await api<{ statuses: GetEvaluationsResponse[] }>(
      `${API_URL}/api/student_answers/${data.test_id}/statuses`
    );
    perStudentStatuses = result.statuses;
    return perStudentStatuses.every((s) => !s.has_any_answer || s.is_all_answers_evaluated);
  }, 5000);

  $effect(() => { if (isPolling) poller.start(); });
  onDestroy(() => poller.stop());

  function scoreColor(status: GetEvaluationsResponse): string {
    if (!status.has_any_answer || !status.is_done_rendering) return 'text-foreground/55';
    const parts = status.total_score.split('/');
    const score = parseInt(parts[0]);
    const max = parseInt(parts[1]);
    if (!max) return 'text-foreground/55';
    const pct = score / max;
    if (pct >= 0.85) return 'text-secondary-700';
    if (pct >= 0.55) return 'text-primary-700';
    return 'text-destructive';
  }

  function iconBg(status: GetEvaluationsResponse): string {
    if (!status.has_any_answer || !status.is_done_rendering) return 'bg-muted';
    const parts = status.total_score.split('/');
    const score = parseInt(parts[0]);
    const max = parseInt(parts[1]);
    if (!max) return 'bg-muted';
    const pct = score / max;
    if (pct >= 0.85) return 'bg-secondary-300';
    if (pct >= 0.55) return 'bg-accent';
    return 'bg-destructive/10';
  }

  function iconColor(status: GetEvaluationsResponse): string {
    if (!status.has_any_answer || !status.is_done_rendering) return 'text-foreground/55';
    const parts = status.total_score.split('/');
    const pct = parseInt(parts[0]) / parseInt(parts[1]);
    if (pct >= 0.85) return 'text-secondary-700';
    if (pct >= 0.55) return 'text-primary-700';
    return 'text-destructive';
  }

  // Stats for summary strip
  const graded = $derived(
    perStudentStatuses.filter((s) => s.has_any_answer && s.is_done_rendering)
  );
  const avgScore = $derived(() => {
    if (!graded.length) return null;
    const scores = graded.map((s) => {
      const p = s.total_score.split('/');
      return { n: parseInt(p[0]), d: parseInt(p[1]) };
    });
    const totalN = scores.reduce((a, s) => a + s.n, 0);
    return `${Math.round(totalN / graded.length)}/${scores[0]?.d ?? '?'}`;
  });
  const pendingCount = $derived(
    perStudentStatuses.filter((s) => !s.has_any_answer).length
  );
</script>

<div class="p-4 flex flex-col gap-2.5">
  <!-- section header -->
  <div class="flex items-center justify-between">
    <span class="font-heading text-[20px] font-extrabold text-foreground tracking-[-0.04em]">Papers</span>
    <button
      onclick={() => { showAddStudent = true; }}
      class="flex items-center gap-1.5 bg-primary text-white text-[13px] font-semibold px-3 py-1.5 rounded-md hover:opacity-90 transition-opacity"
    >
      <IconPlus class="size-4" /> Add
    </button>
  </div>

  <!-- stats strip -->
  {#if perStudentStatuses.length > 0}
    <div class="bg-card rounded-xl shadow-sm flex overflow-hidden">
      {#each [
        { label: 'Graded', val: `${graded.length}/${perStudentStatuses.length}`, color: 'text-secondary-700' },
        { label: 'Avg Score', val: avgScore() ?? '—', color: 'text-primary-700' },
        { label: 'Pending',  val: pendingCount, color: 'text-foreground/55' },
      ] as stat, i}
        <div class="flex-1 py-3 text-center {i < 2 ? 'border-r border-border' : ''}">
          <p class="font-heading text-[20px] font-extrabold tracking-[-0.04em] {stat.color}">{stat.val}</p>
          <p class="text-[10px] font-semibold uppercase tracking-[0.04em] text-foreground/55 mt-0.5">{stat.label}</p>
        </div>
      {/each}
    </div>
  {/if}

  <!-- papers list -->
  {#if perStudentStatuses.length === 0}
    <p class="text-center text-muted-foreground text-sm py-8">No papers yet.</p>
  {:else}
    {#each perStudentStatuses as paper}
      <a
        href="/instances/{data.test_id}/papers/{paper.student_no}"
        class="bg-card rounded-xl shadow-sm px-4 py-3 flex items-center gap-3 no-underline hover:shadow-md transition-shadow"
      >
        <div class="size-[38px] rounded-[10px] flex items-center justify-center shrink-0 {iconBg(paper)}">
          {#if paper.has_any_answer && !paper.is_all_answers_evaluated}
            <Spinner class="size-5 text-primary" />
          {:else if paper.is_done_rendering}
            <IconDone class="size-5 {iconColor(paper)}" />
          {:else}
            <IconUndone class="size-5 text-foreground/55" />
          {/if}
        </div>

        <div class="flex-1 min-w-0">
          <p class="text-[14px] font-medium text-foreground truncate opacity-100">{paper.name}</p>
          <p class="text-[11px] font-mono text-foreground/55 mt-0.5 opacity-100">{paper.student_no}</p>
        </div>

        <div class="flex flex-col items-end gap-1 shrink-0">
          {#if paper.has_any_answer && paper.is_done_rendering}
            <span class="font-heading text-[17px] font-extrabold tracking-[-0.04em] {scoreColor(paper)}">
              {paper.total_score.split('/')[0]}<span class="text-[11px] font-normal text-foreground/55">/{paper.total_score.split('/')[1]}</span>
            </span>
          {:else}
            <span class="font-mono text-[13px] text-foreground/55">—</span>
          {/if}
          <span class="text-[12px] font-heading font-bold px-2 py-0.5 rounded-full
            {paper.is_done_rendering && paper.has_any_answer
              ? (parseInt(paper.total_score) / parseInt(paper.total_score.split('/')[1]) >= 0.85
                  ? 'bg-secondary-300 text-secondary-700'
                  : parseInt(paper.total_score) / parseInt(paper.total_score.split('/')[1]) >= 0.55
                    ? 'bg-accent text-primary-700'
                    : 'bg-destructive/10 text-destructive')
              : paper.has_any_answer && !paper.is_all_answers_evaluated
                ? 'bg-accent text-primary-700'
                : 'bg-muted text-muted-foreground'}">
            {paper.is_done_rendering && paper.has_any_answer
              ? 'Graded'
              : paper.has_any_answer && !paper.is_all_answers_evaluated
                ? 'Grading…'
                : 'Pending'}
          </span>
        </div>
      </a>
    {/each}
  {/if}
</div>

<BottomSheet bind:open={showAddStudent} title="Add Student">
  {#snippet children()}
    <AddNewStudent section_id={data.test_instance?.section_id} close={() => { showAddStudent = false; }} />
  {/snippet}
</BottomSheet>
